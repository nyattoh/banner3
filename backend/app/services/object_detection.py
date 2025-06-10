import cv2
import numpy as np
from ultralytics import YOLO
from rembg import remove, new_session
from typing import List, Tuple, Optional
from PIL import Image
import logging
import torch
from app.models.schemas import ObjectDetectionResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class ObjectDetectionService:
    """Service for detecting objects in images using YOLO and background removal."""
    
    def __init__(self):
        """Initialize the object detection service."""
        self.yolo_model = None
        self.rembg_session = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO model and rembg session."""
        try:
            # Initialize YOLO model
            self.yolo_model = YOLO(settings.yolo_model_path)
            logger.info(f"YOLO model loaded: {settings.yolo_model_path}")
            
            # Initialize rembg session for background removal
            self.rembg_session = new_session('u2net')
            logger.info("Rembg session initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.yolo_model = None
            self.rembg_session = None
    
    def detect_objects(self, image: np.ndarray, method: str = "hybrid") -> ObjectDetectionResult:
        """
        Detect objects in image using specified method.
        
        Args:
            image: Input image as numpy array
            method: Detection method ("yolo", "rembg", "hybrid")
            
        Returns:
            ObjectDetectionResult with detected object regions and metadata
        """
        if method == "yolo":
            return self.detect_objects_yolo(image)
        elif method == "rembg":
            return self.detect_objects_rembg(image)
        elif method == "hybrid":
            return self._detect_objects_hybrid(image)
        else:
            raise ValueError(f"Unsupported detection method: {method}")
    
    def detect_objects_yolo(self, image: np.ndarray) -> ObjectDetectionResult:
        """Detect objects using YOLO model."""
        if self.yolo_model is None:
            logger.warning("YOLO model not available")
            return ObjectDetectionResult(
                object_regions=[],
                confidence_scores=[],
                object_classes=[]
            )
        
        try:
            # Convert BGR to RGB for YOLO
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run YOLO inference
            results = self.yolo_model(rgb_image, verbose=False)
            
            object_regions = []
            confidence_scores = []
            object_classes = []
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates (xyxy format)
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to [x, y, w, h] format
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)
                    
                    # Get confidence and class
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = results[0].names[cls_id]
                    
                    # Filter by confidence threshold
                    if conf > 0.3:  # Confidence threshold
                        object_regions.append([x, y, w, h])
                        confidence_scores.append(conf)
                        object_classes.append(cls_name)
            
            logger.info(f"YOLO detected {len(object_regions)} objects")
            
            return ObjectDetectionResult(
                object_regions=object_regions,
                confidence_scores=confidence_scores,
                object_classes=object_classes
            )
            
        except Exception as e:
            logger.error(f"YOLO object detection failed: {e}")
            return ObjectDetectionResult(
                object_regions=[],
                confidence_scores=[],
                object_classes=[]
            )
    
    def detect_objects_rembg(self, image: np.ndarray) -> ObjectDetectionResult:
        """Detect foreground objects using background removal."""
        if self.rembg_session is None:
            logger.warning("Rembg session not available")
            return ObjectDetectionResult(
                object_regions=[],
                confidence_scores=[],
                object_classes=[]
            )
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Remove background
            result_pil = remove(pil_image, session=self.rembg_session)
            
            # Convert back to numpy array
            result_array = np.array(result_pil)
            
            # Extract alpha channel as mask
            if result_array.shape[2] == 4:
                alpha_mask = result_array[:, :, 3]
            else:
                # If no alpha channel, create mask from non-white areas
                gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
                _, alpha_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            # Find contours of foreground objects
            contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_regions = []
            confidence_scores = []
            object_classes = []
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Filter by minimum area
                if area > 100:  # Minimum area threshold
                    object_regions.append([x, y, w, h])
                    # Use contour area as a proxy for confidence
                    confidence = min(1.0, area / (image.shape[0] * image.shape[1]))
                    confidence_scores.append(confidence)
                    object_classes.append("foreground_object")
            
            logger.info(f"Rembg detected {len(object_regions)} foreground objects")
            
            return ObjectDetectionResult(
                object_regions=object_regions,
                confidence_scores=confidence_scores,
                object_classes=object_classes
            )
            
        except Exception as e:
            logger.error(f"Rembg object detection failed: {e}")
            return ObjectDetectionResult(
                object_regions=[],
                confidence_scores=[],
                object_classes=[]
            )
    
    def _detect_objects_hybrid(self, image: np.ndarray) -> ObjectDetectionResult:
        """Combine YOLO and rembg results for better detection."""
        yolo_result = self.detect_objects_yolo(image)
        rembg_result = self.detect_objects_rembg(image)
        
        # Merge results
        combined_regions = []
        combined_scores = []
        combined_classes = []
        
        # Add YOLO results first (more specific classes)
        for i, region in enumerate(yolo_result.object_regions):
            combined_regions.append(region)
            combined_scores.append(yolo_result.confidence_scores[i])
            combined_classes.append(yolo_result.object_classes[i])
        
        # Add rembg results that don't overlap significantly with YOLO results
        for i, region in enumerate(rembg_result.object_regions):
            if not self._has_significant_overlap(region, combined_regions):
                combined_regions.append(region)
                combined_scores.append(rembg_result.confidence_scores[i])
                combined_classes.append(rembg_result.object_classes[i])
        
        logger.info(f"Hybrid detection found {len(combined_regions)} objects")
        
        return ObjectDetectionResult(
            object_regions=combined_regions,
            confidence_scores=combined_scores,
            object_classes=combined_classes
        )
    
    def _has_significant_overlap(self, region: List[int], existing_regions: List[List[int]], 
                               threshold: float = 0.3) -> bool:
        """Check if a region overlaps significantly with existing regions."""
        x1, y1, w1, h1 = region
        
        for x2, y2, w2, h2 in existing_regions:
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                if iou > threshold:
                    return True
        
        return False
    
    def generate_object_mask(self, image: np.ndarray, 
                           detection_result: ObjectDetectionResult) -> np.ndarray:
        """
        Generate a binary mask for detected object regions.
        
        Args:
            image: Original image
            detection_result: Result from object detection
            
        Returns:
            Binary mask where object regions are white (255) and background is black (0)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in detection_result.object_regions:
            x, y, w, h = region
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1]))
            y = max(0, min(y, image.shape[0]))
            w = max(0, min(w, image.shape[1] - x))
            h = max(0, min(h, image.shape[0] - y))
            
            if w > 0 and h > 0:
                mask[y:y+h, x:x+w] = 255
        
        return mask
    
    def generate_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate foreground mask using background removal."""
        if self.rembg_session is None:
            logger.warning("Rembg session not available, returning empty mask")
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Remove background
            result_pil = remove(pil_image, session=self.rembg_session)
            result_array = np.array(result_pil)
            
            # Extract alpha channel as mask
            if result_array.shape[2] == 4:
                mask = result_array[:, :, 3]
            else:
                gray = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            logger.error(f"Foreground mask generation failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def filter_by_confidence(self, detection_result: ObjectDetectionResult, 
                           threshold: float = 0.5) -> ObjectDetectionResult:
        """Filter detections by confidence threshold."""
        filtered_regions = []
        filtered_scores = []
        filtered_classes = []
        
        for i, score in enumerate(detection_result.confidence_scores):
            if score >= threshold:
                filtered_regions.append(detection_result.object_regions[i])
                filtered_scores.append(score)
                filtered_classes.append(detection_result.object_classes[i])
        
        return ObjectDetectionResult(
            object_regions=filtered_regions,
            confidence_scores=filtered_scores,
            object_classes=filtered_classes
        )
    
    def filter_by_size(self, detection_result: ObjectDetectionResult, 
                      min_area: int = 100) -> ObjectDetectionResult:
        """Filter detections by minimum area."""
        filtered_regions = []
        filtered_scores = []
        filtered_classes = []
        
        for i, region in enumerate(detection_result.object_regions):
            x, y, w, h = region
            area = w * h
            
            if area >= min_area:
                filtered_regions.append(region)
                filtered_scores.append(detection_result.confidence_scores[i])
                filtered_classes.append(detection_result.object_classes[i])
        
        return ObjectDetectionResult(
            object_regions=filtered_regions,
            confidence_scores=filtered_scores,
            object_classes=filtered_classes
        )
    
    def handle_overlapping_objects(self, detection_result: ObjectDetectionResult, 
                                 iou_threshold: float = 0.5) -> ObjectDetectionResult:
        """Handle overlapping detections using Non-Maximum Suppression."""
        if len(detection_result.object_regions) == 0:
            return detection_result
        
        # Convert to format expected by NMS
        boxes = []
        scores = []
        
        for i, region in enumerate(detection_result.object_regions):
            x, y, w, h = region
            boxes.append([x, y, x + w, y + h])  # Convert to xyxy format
            scores.append(detection_result.confidence_scores[i])
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            score_threshold=0.3, 
            nms_threshold=iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            
            filtered_regions = []
            filtered_scores = []
            filtered_classes = []
            
            for i in indices:
                # Convert back to xywh format
                x1, y1, x2, y2 = boxes[i]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                filtered_regions.append([x, y, w, h])
                filtered_scores.append(detection_result.confidence_scores[i])
                filtered_classes.append(detection_result.object_classes[i])
            
            return ObjectDetectionResult(
                object_regions=filtered_regions,
                confidence_scores=filtered_scores,
                object_classes=filtered_classes
            )
        
        return ObjectDetectionResult(
            object_regions=[],
            confidence_scores=[],
            object_classes=[]
        )
    
    def validate_object_regions(self, detection_result: ObjectDetectionResult) -> bool:
        """Validate that object regions have valid coordinates."""
        for region in detection_result.object_regions:
            x, y, w, h = region
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError(f"Invalid object region coordinates: {region}")
        return True
    
    def extract_object_layer(self, image: np.ndarray, 
                           detection_result: ObjectDetectionResult) -> np.ndarray:
        """
        Extract object layer as transparent PNG.
        
        Args:
            image: Original image
            detection_result: Object detection results
            
        Returns:
            RGBA image with objects preserved and background transparent
        """
        # Create RGBA output
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Generate object mask
        object_mask = self.generate_object_mask(image, detection_result)
        
        # Copy original image colors where objects are detected
        if len(image.shape) == 3:
            result[:, :, :3] = image
        else:
            result[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Set alpha channel based on object mask
        result[:, :, 3] = object_mask
        
        return result