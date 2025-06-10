import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image
import logging
from sklearn.cluster import KMeans
from scipy import ndimage
import skimage.filters
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)


class BackgroundInpaintingService:
    """Service for inpainting background regions where text and objects are removed."""
    
    def __init__(self):
        """Initialize the background inpainting service."""
        self.inpaint_radius = 3
        self.default_method = 'hybrid'
    
    def generate_background_layer(self, image: np.ndarray, text_mask: np.ndarray, 
                                object_mask: np.ndarray) -> np.ndarray:
        """
        Generate complete background layer by inpainting removed text and objects.
        
        Args:
            image: Original image
            text_mask: Binary mask of text regions (white=text, black=background)
            object_mask: Binary mask of object regions (white=object, black=background)
            
        Returns:
            Background layer with text and objects inpainted
        """
        try:
            # Validate inputs
            if text_mask.shape[:2] != image.shape[:2] or object_mask.shape[:2] != image.shape[:2]:
                raise ValueError("Mask dimensions must match image dimensions")
            
            # Combine text and object masks
            combined_mask = self.create_combined_mask(text_mask, object_mask)
            
            # Apply inpainting
            background_layer = self.inpaint_background_hybrid(image, combined_mask)
            
            logger.info("Background layer generated successfully")
            return background_layer
            
        except Exception as e:
            logger.error(f"Background layer generation failed: {e}")
            return image.copy()  # Return original image as fallback
    
    def create_combined_mask(self, text_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
        """Combine text and object masks into a single inpainting mask."""
        # Ensure masks are binary
        text_binary = (text_mask > 128).astype(np.uint8) * 255
        object_binary = (object_mask > 128).astype(np.uint8) * 255
        
        # Combine using logical OR
        combined = cv2.bitwise_or(text_binary, object_binary)
        
        # Apply slight dilation to ensure good inpainting boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, kernel)
        
        return combined
    
    def inpaint_background_opencv(self, image: np.ndarray, mask: np.ndarray, 
                                method: str = 'ns') -> np.ndarray:
        """
        Inpaint background using OpenCV methods.
        
        Args:
            image: Original image
            mask: Binary mask of regions to inpaint
            method: 'ns' for Navier-Stokes, 'telea' for Telea algorithm
            
        Returns:
            Inpainted image
        """
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("Mask dimensions must match image dimensions")
        
        # Calculate adaptive radius based on mask size
        radius = self.calculate_adaptive_radius(mask)
        
        if method == 'ns':
            inpaint_method = cv2.INPAINT_NS
        elif method == 'telea':
            inpaint_method = cv2.INPAINT_TELEA
        else:
            raise ValueError(f"Unsupported inpainting method: {method}")
        
        try:
            result = cv2.inpaint(image, mask, radius, inpaint_method)
            return result
        except Exception as e:
            logger.error(f"OpenCV inpainting failed: {e}")
            return image.copy()
    
    def inpaint_background_hybrid(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Hybrid inpainting combining multiple techniques for best results.
        
        Args:
            image: Original image
            mask: Binary mask of regions to inpaint
            
        Returns:
            Inpainted image using hybrid approach
        """
        try:
            # Start with Telea method for structure
            telea_result = self.inpaint_background_opencv(image, mask, 'telea')
            
            # Refine with Navier-Stokes for texture
            ns_result = self.inpaint_background_opencv(telea_result, mask, 'ns')
            
            # Apply edge-preserving smoothing to reduce artifacts
            refined_result = self.inpaint_edge_preserving(ns_result, mask)
            
            # Final texture synthesis for large areas
            if np.sum(mask) > (mask.shape[0] * mask.shape[1] * 0.1):  # If mask covers >10%
                final_result = self.inpaint_texture_synthesis(refined_result, mask)
            else:
                final_result = refined_result
            
            return final_result
            
        except Exception as e:
            logger.error(f"Hybrid inpainting failed: {e}")
            return self.inpaint_background_opencv(image, mask, 'telea')
    
    def inpaint_edge_preserving(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply edge-preserving inpainting to reduce artifacts."""
        try:
            # Apply bilateral filter to preserve edges
            filtered = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Blend with original based on mask
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            result = filtered * mask_3d + image * (1 - mask_3d)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Edge-preserving inpainting failed: {e}")
            return image.copy()
    
    def inpaint_texture_synthesis(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using texture synthesis for natural-looking results."""
        try:
            # Detect background patterns
            pattern_info = self.detect_background_patterns(image)
            
            # Apply texture-based inpainting
            result = self.inpaint_with_pattern(image, mask, pattern_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Texture synthesis inpainting failed: {e}")
            return image.copy()
    
    def inpaint_seamless_cloning(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use seamless cloning for natural inpainting."""
        try:
            # Find a good source region for cloning
            source_region = self.find_source_region(image, mask)
            
            # Apply seamless cloning
            mask_center = self.get_mask_center(mask)
            
            result = cv2.seamlessClone(
                source_region, image, mask, mask_center, cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Seamless cloning failed: {e}")
            return image.copy()
    
    def inpaint_progressive(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Progressive inpainting for large areas."""
        try:
            result = image.copy()
            
            # Erode mask progressively and inpaint in layers
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            current_mask = mask.copy()
            
            for _ in range(3):  # Multiple iterations
                # Inpaint current layer
                result = cv2.inpaint(result, current_mask, 3, cv2.INPAINT_TELEA)
                
                # Erode mask for next iteration
                current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_ERODE, kernel)
                
                if np.sum(current_mask) == 0:  # No more pixels to inpaint
                    break
            
            return result
            
        except Exception as e:
            logger.error(f"Progressive inpainting failed: {e}")
            return image.copy()
    
    def inpaint_context_aware(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Context-aware inpainting considering surrounding regions."""
        try:
            # Analyze context around masked regions
            context_info = self.analyze_context(image, mask)
            
            # Apply context-guided inpainting
            result = self.apply_context_guided_inpainting(image, mask, context_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Context-aware inpainting failed: {e}")
            return self.inpaint_background_opencv(image, mask, 'telea')
    
    def inpaint_multiscale(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Multi-scale inpainting approach."""
        try:
            # Create image pyramid
            pyramid_levels = 3
            result = image.copy()
            
            for level in range(pyramid_levels):
                # Downsample
                scale = 2 ** level
                small_image = cv2.resize(result, 
                                       (result.shape[1] // scale, result.shape[0] // scale))
                small_mask = cv2.resize(mask, 
                                      (mask.shape[1] // scale, mask.shape[0] // scale))
                
                # Inpaint at this scale
                small_inpainted = cv2.inpaint(small_image, small_mask, 3, cv2.INPAINT_TELEA)
                
                # Upsample and blend
                upsampled = cv2.resize(small_inpainted, (result.shape[1], result.shape[0]))
                
                # Blend based on mask
                mask_norm = mask.astype(np.float32) / 255.0
                if len(result.shape) == 3:
                    mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
                
                result = (upsampled * mask_norm + result * (1 - mask_norm)).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-scale inpainting failed: {e}")
            return image.copy()
    
    def detect_background_patterns(self, image: np.ndarray) -> Dict:
        """Detect background patterns for texture synthesis."""
        try:
            # Convert to LAB color space for better color analysis
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Detect dominant colors
            pixels = lab_image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_
            
            # Analyze texture using LBP
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=10)
            
            # Calculate texture features
            texture_features = {
                'lbp_histogram': lbp_hist,
                'mean_intensity': np.mean(gray),
                'std_intensity': np.std(gray)
            }
            
            return {
                'dominant_colors': dominant_colors,
                'texture_features': texture_features
            }
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {'dominant_colors': [], 'texture_features': {}}
    
    def inpaint_with_pattern(self, image: np.ndarray, mask: np.ndarray, 
                           pattern_info: Dict) -> np.ndarray:
        """Inpaint using detected patterns."""
        try:
            result = image.copy()
            
            # Use dominant color for initial fill
            if pattern_info['dominant_colors'].size > 0:
                # Convert back to BGR
                dominant_lab = pattern_info['dominant_colors'][0]
                # Create a simple pattern-based fill
                mask_indices = np.where(mask > 0)
                
                # Apply slight variations based on texture features
                for i, (y, x) in enumerate(zip(mask_indices[0], mask_indices[1])):
                    # Add some variation based on position
                    variation = np.sin(x * 0.1) * np.cos(y * 0.1) * 10
                    color = dominant_lab + variation
                    color = np.clip(color, 0, 255)
                    result[y, x] = color
            
            # Apply smoothing to blend with surroundings
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_dilated = cv2.dilate(mask, kernel)
            
            # Gaussian blur on the inpainted region
            blurred = cv2.GaussianBlur(result, (15, 15), 0)
            mask_norm = mask_dilated.astype(np.float32) / 255.0
            
            if len(result.shape) == 3:
                mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
            
            result = (blurred * mask_norm + result * (1 - mask_norm)).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern-based inpainting failed: {e}")
            return image.copy()
    
    def analyze_context(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze context around masked regions."""
        try:
            # Create a border around the mask to analyze context
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask_dilated = cv2.dilate(mask, kernel)
            context_mask = mask_dilated - mask  # Ring around the mask
            
            # Extract context features
            context_pixels = image[context_mask > 0]
            
            if len(context_pixels) > 0:
                context_info = {
                    'mean_color': np.mean(context_pixels, axis=0),
                    'std_color': np.std(context_pixels, axis=0),
                    'median_color': np.median(context_pixels, axis=0)
                }
            else:
                context_info = {
                    'mean_color': np.mean(image, axis=(0, 1)),
                    'std_color': np.std(image, axis=(0, 1)),
                    'median_color': np.median(image, axis=(0, 1))
                }
            
            return context_info
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {'mean_color': [128, 128, 128], 'std_color': [0, 0, 0], 'median_color': [128, 128, 128]}
    
    def apply_context_guided_inpainting(self, image: np.ndarray, mask: np.ndarray, 
                                      context_info: Dict) -> np.ndarray:
        """Apply inpainting guided by context information."""
        try:
            # Start with standard inpainting
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            
            # Adjust colors based on context
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                # Blend with context colors
                context_color = context_info['mean_color']
                
                for y, x in zip(mask_indices[0], mask_indices[1]):
                    current_color = result[y, x]
                    # Weighted blend with context
                    blended_color = 0.7 * current_color + 0.3 * context_color
                    result[y, x] = np.clip(blended_color, 0, 255)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Context-guided inpainting failed: {e}")
            return image.copy()
    
    def calculate_adaptive_radius(self, mask: np.ndarray) -> int:
        """Calculate adaptive inpainting radius based on mask characteristics."""
        # Calculate the size of the masked regions
        mask_area = np.sum(mask > 0)
        total_area = mask.shape[0] * mask.shape[1]
        mask_ratio = mask_area / total_area
        
        # Adaptive radius based on mask size
        if mask_ratio < 0.01:  # Small mask
            return 2
        elif mask_ratio < 0.05:  # Medium mask
            return 3
        elif mask_ratio < 0.15:  # Large mask
            return 5
        else:  # Very large mask
            return 7
    
    def dilate_mask(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Dilate mask to expand inpainting regions."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    def erode_mask(self, mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Erode mask to shrink inpainting regions."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    
    def assess_inpainting_quality(self, original: np.ndarray, inpainted: np.ndarray, 
                                mask: np.ndarray) -> float:
        """Assess the quality of inpainting result."""
        try:
            # Calculate structural similarity in non-masked regions
            non_mask = (mask == 0)
            
            if np.sum(non_mask) == 0:
                return 0.0
            
            # Compare gradients in boundary regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            boundary = cv2.dilate(mask, kernel) - mask
            
            if np.sum(boundary) > 0:
                orig_boundary = original[boundary > 0]
                inp_boundary = inpainted[boundary > 0]
                
                # Calculate color difference
                color_diff = np.mean(np.abs(orig_boundary.astype(np.float32) - 
                                          inp_boundary.astype(np.float32)))
                
                # Normalize to 0-1 score (lower difference = higher quality)
                quality_score = max(0.0, 1.0 - color_diff / 255.0)
            else:
                quality_score = 1.0
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.5  # Default medium quality
    
    def find_source_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Find a good source region for seamless cloning."""
        try:
            # Find non-masked regions that are similar to mask boundary
            inverted_mask = cv2.bitwise_not(mask)
            
            # Use the largest connected component as source
            contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                source_mask = np.zeros_like(mask)
                cv2.fillPoly(source_mask, [largest_contour], 255)
                
                # Extract source region
                source_region = image.copy()
                source_region[source_mask == 0] = 0
                
                return source_region
            
            return image.copy()
            
        except Exception as e:
            logger.error(f"Source region finding failed: {e}")
            return image.copy()
    
    def get_mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get the center point of the mask for seamless cloning."""
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        else:
            return (mask.shape[1] // 2, mask.shape[0] // 2)