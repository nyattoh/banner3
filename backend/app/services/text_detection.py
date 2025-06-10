import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Tuple, Optional
from PIL import Image
import logging
from app.models.schemas import TextDetectionResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class TextDetectionService:
    """Service for detecting text in images using multiple OCR engines."""
    
    def __init__(self):
        """Initialize the text detection service."""
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        self.easyocr_reader = None
        self._initialize_easyocr()
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR reader with Japanese and English support."""
        try:
            self.easyocr_reader = easyocr.Reader(['en', 'ja'], gpu=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    def detect_text(self, image: np.ndarray, engine: str = "hybrid") -> TextDetectionResult:
        """
        Detect text in image using specified engine.
        
        Args:
            image: Input image as numpy array
            engine: OCR engine to use ("tesseract", "easyocr", "hybrid")
            
        Returns:
            TextDetectionResult with detected text regions and metadata
        """
        if engine == "tesseract":
            return self.detect_text_tesseract(image)
        elif engine == "easyocr":
            return self.detect_text_easyocr(image)
        elif engine == "hybrid":
            return self._detect_text_hybrid(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def detect_text_tesseract(self, image: np.ndarray) -> TextDetectionResult:
        """Detect text using Tesseract OCR."""
        try:
            # Convert to PIL Image for Tesseract
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                pil_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            text_regions = []
            confidence_scores = []
            detected_text = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                # Filter out low confidence and empty detections
                if conf > 30 and text:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text_regions.append([x, y, w, h])
                    confidence_scores.append(conf / 100.0)  # Normalize to 0-1
                    detected_text.append(text)
            
            logger.info(f"Tesseract detected {len(text_regions)} text regions")
            
            return TextDetectionResult(
                text_regions=text_regions,
                confidence_scores=confidence_scores,
                detected_text=detected_text
            )
            
        except Exception as e:
            logger.error(f"Tesseract text detection failed: {e}")
            return TextDetectionResult(
                text_regions=[],
                confidence_scores=[],
                detected_text=[]
            )
    
    def detect_text_easyocr(self, image: np.ndarray) -> TextDetectionResult:
        """Detect text using EasyOCR."""
        if self.easyocr_reader is None:
            logger.warning("EasyOCR not available, falling back to empty result")
            return TextDetectionResult(
                text_regions=[],
                confidence_scores=[],
                detected_text=[]
            )
        
        try:
            # EasyOCR expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect text
            results = self.easyocr_reader.readtext(rgb_image)
            
            text_regions = []
            confidence_scores = []
            detected_text = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Confidence threshold
                    # Convert bbox to [x, y, w, h] format
                    bbox_array = np.array(bbox)
                    x_min, y_min = bbox_array.min(axis=0)
                    x_max, y_max = bbox_array.max(axis=0)
                    
                    x, y = int(x_min), int(y_min)
                    w, h = int(x_max - x_min), int(y_max - y_min)
                    
                    text_regions.append([x, y, w, h])
                    confidence_scores.append(float(confidence))
                    detected_text.append(text.strip())
            
            logger.info(f"EasyOCR detected {len(text_regions)} text regions")
            
            return TextDetectionResult(
                text_regions=text_regions,
                confidence_scores=confidence_scores,
                detected_text=detected_text
            )
            
        except Exception as e:
            logger.error(f"EasyOCR text detection failed: {e}")
            return TextDetectionResult(
                text_regions=[],
                confidence_scores=[],
                detected_text=[]
            )
    
    def _detect_text_hybrid(self, image: np.ndarray) -> TextDetectionResult:
        """Combine results from multiple OCR engines for better accuracy."""
        tesseract_result = self.detect_text_tesseract(image)
        easyocr_result = self.detect_text_easyocr(image)
        
        # Merge results and remove duplicates
        combined_regions = []
        combined_scores = []
        combined_text = []
        
        # Add Tesseract results
        for i, region in enumerate(tesseract_result.text_regions):
            combined_regions.append(region)
            combined_scores.append(tesseract_result.confidence_scores[i])
            combined_text.append(tesseract_result.detected_text[i])
        
        # Add EasyOCR results that don't overlap significantly
        for i, region in enumerate(easyocr_result.text_regions):
            if not self._has_significant_overlap(region, combined_regions):
                combined_regions.append(region)
                combined_scores.append(easyocr_result.confidence_scores[i])
                combined_text.append(easyocr_result.detected_text[i])
        
        logger.info(f"Hybrid detection found {len(combined_regions)} text regions")
        
        return TextDetectionResult(
            text_regions=combined_regions,
            confidence_scores=combined_scores,
            detected_text=combined_text
        )
    
    def _has_significant_overlap(self, region: List[int], existing_regions: List[List[int]], 
                               threshold: float = 0.5) -> bool:
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
    
    def generate_text_mask(self, image: np.ndarray, 
                          detection_result: TextDetectionResult) -> np.ndarray:
        """
        Generate a binary mask for detected text regions.
        
        Args:
            image: Original image
            detection_result: Result from text detection
            
        Returns:
            Binary mask where text regions are white (255) and background is black (0)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in detection_result.text_regions:
            x, y, w, h = region
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1]))
            y = max(0, min(y, image.shape[0]))
            w = max(0, min(w, image.shape[1] - x))
            h = max(0, min(h, image.shape[0] - y))
            
            if w > 0 and h > 0:
                mask[y:y+h, x:x+w] = 255
        
        return mask
    
    def validate_text_regions(self, detection_result: TextDetectionResult) -> bool:
        """Validate that text regions have valid coordinates."""
        for region in detection_result.text_regions:
            x, y, w, h = region
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError(f"Invalid text region coordinates: {region}")
        return True
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve OCR accuracy."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply image enhancement techniques
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. Binarization (Otsu's thresholding)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_layer(self, image: np.ndarray, 
                          detection_result: TextDetectionResult) -> np.ndarray:
        """
        Extract text layer as transparent PNG.
        
        Args:
            image: Original image
            detection_result: Text detection results
            
        Returns:
            RGBA image with text preserved and background transparent
        """
        # Create RGBA output
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Generate text mask
        text_mask = self.generate_text_mask(image, detection_result)
        
        # Copy original image colors where text is detected
        if len(image.shape) == 3:
            result[:, :, :3] = image
        else:
            result[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Set alpha channel based on text mask
        result[:, :, 3] = text_mask
        
        return result