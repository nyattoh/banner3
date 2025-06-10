import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from app.models.schemas import LayerCompositionValidation

logger = logging.getLogger(__name__)


class LayerCompositionService:
    """Service for composing and validating layer reconstruction."""
    
    def __init__(self):
        """Initialize the layer composition service."""
        self.similarity_threshold = 0.85
        self.acceptable_error_threshold = 0.05
    
    def compose_layers(self, background: np.ndarray, text_layer: np.ndarray, 
                      object_layer: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compose layers to reconstruct the original image.
        
        Args:
            background: Background layer (RGB)
            text_layer: Text layer with alpha channel (RGBA)
            object_layer: Optional object layer with alpha channel (RGBA)
            
        Returns:
            Composed image (RGB)
        """
        try:
            # Validate dimensions
            self._validate_layer_dimensions(background, text_layer, object_layer)
            
            # Start with background
            result = background.copy().astype(np.float32)
            
            # Compose text layer
            result = self._blend_layer_alpha(result, text_layer)
            
            # Compose object layer if provided
            if object_layer is not None:
                result = self._blend_layer_alpha(result, object_layer)
            
            # Convert back to uint8
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            logger.info("Layers composed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Layer composition failed: {e}")
            return background.copy()
    
    def _validate_layer_dimensions(self, background: np.ndarray, text_layer: np.ndarray, 
                                 object_layer: Optional[np.ndarray] = None):
        """Validate that all layers have compatible dimensions."""
        base_shape = background.shape[:2]
        
        if text_layer.shape[:2] != base_shape:
            raise ValueError(f"Text layer dimensions {text_layer.shape[:2]} don't match background {base_shape}")
        
        if object_layer is not None and object_layer.shape[:2] != base_shape:
            raise ValueError(f"Object layer dimensions {object_layer.shape[:2]} don't match background {base_shape}")
        
        # Check channel requirements
        if len(background.shape) != 3 or background.shape[2] != 3:
            raise ValueError("Background layer must be RGB (3 channels)")
        
        if len(text_layer.shape) != 3 or text_layer.shape[2] != 4:
            raise ValueError("Text layer must be RGBA (4 channels)")
        
        if object_layer is not None and (len(object_layer.shape) != 3 or object_layer.shape[2] != 4):
            raise ValueError("Object layer must be RGBA (4 channels)")
    
    def _blend_layer_alpha(self, base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Blend overlay layer with base using alpha channel."""
        if overlay.shape[2] != 4:
            return base
        
        # Extract alpha channel and normalize
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        # Extract RGB channels from overlay
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        
        # Alpha blending
        result = overlay_rgb * alpha + base * (1 - alpha)
        
        return result
    
    def validate_composition(self, original: np.ndarray, composed: np.ndarray) -> LayerCompositionValidation:
        """
        Validate that the composed layers accurately reconstruct the original image.
        
        Args:
            original: Original input image
            composed: Composed result from layers
            
        Returns:
            LayerCompositionValidation with validation results
        """
        try:
            # Calculate similarity score
            similarity_score = self.calculate_similarity(original, composed)
            
            # Determine if composition is valid
            is_valid = similarity_score >= self.similarity_threshold
            
            # Detect specific errors
            errors = self.detect_composition_errors(original, composed)
            
            # Add error for low similarity
            if not is_valid:
                errors.append(f"Low similarity score: {similarity_score:.3f} < {self.similarity_threshold}")
            
            logger.info(f"Composition validation: valid={is_valid}, similarity={similarity_score:.3f}")
            
            return LayerCompositionValidation(
                is_valid=is_valid,
                similarity_score=float(similarity_score),
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Composition validation failed: {e}")
            return LayerCompositionValidation(
                is_valid=False,
                similarity_score=0.0,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def calculate_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate similarity between two images using multiple metrics."""
        try:
            # Ensure images have same dimensions
            if image1.shape != image2.shape:
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            
            # Convert to grayscale for SSIM calculation
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = image1
                gray2 = image2
            
            # Calculate SSIM
            ssim_score = ssim(gray1, gray2, data_range=255)
            
            # Calculate normalized correlation coefficient
            norm_corr = self._normalized_correlation(image1, image2)
            
            # Combine metrics (weighted average)
            combined_score = 0.7 * ssim_score + 0.3 * norm_corr
            
            return float(np.clip(combined_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _normalized_correlation(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate normalized correlation coefficient."""
        try:
            # Flatten images
            flat1 = image1.flatten().astype(np.float64)
            flat2 = image2.flatten().astype(np.float64)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return 0.0
    
    def detect_composition_errors(self, original: np.ndarray, composed: np.ndarray) -> List[str]:
        """Detect specific types of composition errors."""
        errors = []
        
        try:
            # Check for color artifacts
            color_diff = cv2.absdiff(original, composed)
            mean_color_diff = np.mean(color_diff)
            
            if mean_color_diff > 30:  # Threshold for significant color difference
                errors.append(f"Significant color differences detected (mean diff: {mean_color_diff:.1f})")
            
            # Check for structural artifacts
            structural_error = self._detect_structural_errors(original, composed)
            if structural_error:
                errors.append(structural_error)
            
            # Check for edge preservation
            edge_error = self._check_edge_preservation(original, composed)
            if edge_error:
                errors.append(edge_error)
            
            # Check for texture consistency
            texture_error = self._check_texture_consistency(original, composed)
            if texture_error:
                errors.append(texture_error)
            
        except Exception as e:
            errors.append(f"Error detection failed: {str(e)}")
        
        return errors
    
    def _detect_structural_errors(self, original: np.ndarray, composed: np.ndarray) -> Optional[str]:
        """Detect structural inconsistencies."""
        try:
            # Compare edge maps
            orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
            comp_edges = cv2.Canny(cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY), 50, 150)
            
            edge_diff = cv2.absdiff(orig_edges, comp_edges)
            edge_error_ratio = np.sum(edge_diff > 0) / edge_diff.size
            
            if edge_error_ratio > 0.1:  # More than 10% edge differences
                return f"Structural inconsistencies detected (edge diff: {edge_error_ratio:.3f})"
            
            return None
            
        except Exception as e:
            return f"Structural error detection failed: {str(e)}"
    
    def _check_edge_preservation(self, original: np.ndarray, composed: np.ndarray) -> Optional[str]:
        """Check if edges are properly preserved."""
        try:
            # Use Sobel operator for edge detection
            orig_sobel = cv2.Sobel(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1, ksize=3)
            comp_sobel = cv2.Sobel(cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 1, ksize=3)
            
            # Calculate correlation between edge maps
            edge_correlation = np.corrcoef(orig_sobel.flatten(), comp_sobel.flatten())[0, 1]
            
            if edge_correlation < 0.8:  # Low edge correlation
                return f"Poor edge preservation (correlation: {edge_correlation:.3f})"
            
            return None
            
        except Exception as e:
            return f"Edge preservation check failed: {str(e)}"
    
    def _check_texture_consistency(self, original: np.ndarray, composed: np.ndarray) -> Optional[str]:
        """Check texture consistency between images."""
        try:
            # Calculate local binary patterns for texture analysis
            from skimage.feature import local_binary_pattern
            
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            comp_gray = cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY)
            
            # Compute LBP
            orig_lbp = local_binary_pattern(orig_gray, 8, 1, method='uniform')
            comp_lbp = local_binary_pattern(comp_gray, 8, 1, method='uniform')
            
            # Compare LBP histograms
            orig_hist, _ = np.histogram(orig_lbp, bins=10, range=(0, 10))
            comp_hist, _ = np.histogram(comp_lbp, bins=10, range=(0, 10))
            
            # Normalize histograms
            orig_hist = orig_hist.astype(np.float32) / np.sum(orig_hist)
            comp_hist = comp_hist.astype(np.float32) / np.sum(comp_hist)
            
            # Calculate histogram correlation
            texture_correlation = np.corrcoef(orig_hist, comp_hist)[0, 1]
            
            if texture_correlation < 0.7:  # Low texture correlation
                return f"Texture inconsistency detected (correlation: {texture_correlation:.3f})"
            
            return None
            
        except Exception as e:
            return f"Texture consistency check failed: {str(e)}"
    
    def check_layer_completeness(self, original: np.ndarray, text_layer: np.ndarray, 
                               object_layer: np.ndarray, background_layer: np.ndarray) -> Dict:
        """Check if layers completely cover the original image."""
        try:
            # Create combined mask from text and object layers
            text_mask = (text_layer[:, :, 3] > 0).astype(np.uint8)
            object_mask = (object_layer[:, :, 3] > 0).astype(np.uint8)
            
            # Combined foreground mask
            foreground_mask = np.logical_or(text_mask, object_mask)
            
            # Calculate coverage
            total_pixels = original.shape[0] * original.shape[1]
            covered_pixels = np.sum(foreground_mask) + np.sum(~foreground_mask)  # Always 100% conceptually
            
            # Check for missing regions (where no layer is dominant)
            missing_regions = self._find_missing_regions(original, text_layer, object_layer, background_layer)
            
            coverage_percentage = 100.0  # Layers always cover 100% by definition
            
            return {
                'coverage_percentage': coverage_percentage,
                'missing_regions': missing_regions,
                'text_coverage': float(np.sum(text_mask) / total_pixels * 100),
                'object_coverage': float(np.sum(object_mask) / total_pixels * 100),
                'background_coverage': float(np.sum(~foreground_mask) / total_pixels * 100)
            }
            
        except Exception as e:
            logger.error(f"Layer completeness check failed: {e}")
            return {'coverage_percentage': 0.0, 'missing_regions': [], 'error': str(e)}
    
    def _find_missing_regions(self, original: np.ndarray, text_layer: np.ndarray, 
                            object_layer: np.ndarray, background_layer: np.ndarray) -> List[Dict]:
        """Find regions where layers don't adequately represent the original."""
        missing_regions = []
        
        try:
            # Compose layers
            composed = self.compose_layers(background_layer, text_layer, object_layer)
            
            # Find regions with high difference
            diff = cv2.absdiff(original, composed)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find problematic regions
            _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours of problematic regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    missing_regions.append({
                        'bbox': [x, y, w, h],
                        'area': float(area),
                        'mean_error': float(np.mean(gray_diff[y:y+h, x:x+w]))
                    })
            
        except Exception as e:
            logger.error(f"Missing regions detection failed: {e}")
        
        return missing_regions
    
    def blend_layers_with_mask(self, layer1: np.ndarray, layer2: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """Blend two layers using a custom mask."""
        try:
            # Normalize mask to [0, 1]
            mask_norm = mask.astype(np.float32) / 255.0
            
            # Expand mask to match layer channels
            if len(layer1.shape) == 3:
                mask_norm = np.expand_dims(mask_norm, axis=2)
                mask_norm = np.repeat(mask_norm, layer1.shape[2], axis=2)
            
            # Blend layers
            blended = layer1.astype(np.float32) * mask_norm + layer2.astype(np.float32) * (1 - mask_norm)
            
            return np.clip(blended, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Layer blending with mask failed: {e}")
            return layer1.copy()
    
    def extract_layer_differences(self, original: np.ndarray, composed: np.ndarray) -> np.ndarray:
        """Extract difference map between original and composed images."""
        try:
            # Calculate absolute difference
            diff = cv2.absdiff(original, composed)
            
            # Convert to grayscale for visualization
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Apply color map for better visualization
            colored_diff = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
            
            return colored_diff
            
        except Exception as e:
            logger.error(f"Difference extraction failed: {e}")
            return np.zeros_like(original)
    
    def optimize_layer_composition(self, text_layer: np.ndarray, object_layer: np.ndarray, 
                                 background_layer: np.ndarray, target: np.ndarray) -> Dict:
        """Optimize layer composition to better match target image."""
        try:
            # This is a simplified optimization - in practice, could use more sophisticated methods
            optimized_layers = {
                'text_layer': text_layer.copy(),
                'object_layer': object_layer.copy(),
                'background_layer': background_layer.copy()
            }
            
            # Iterative refinement
            best_score = 0.0
            
            for iteration in range(5):  # Limited iterations for performance
                # Compose current layers
                composed = self.compose_layers(
                    optimized_layers['background_layer'],
                    optimized_layers['text_layer'],
                    optimized_layers['object_layer']
                )
                
                # Calculate current score
                current_score = self.calculate_similarity(target, composed)
                
                if current_score > best_score:
                    best_score = current_score
                
                # Simple optimization: adjust alpha channels based on error
                self._refine_layer_alpha(optimized_layers, target, composed)
            
            logger.info(f"Layer optimization completed. Best score: {best_score:.3f}")
            return optimized_layers
            
        except Exception as e:
            logger.error(f"Layer optimization failed: {e}")
            return {
                'text_layer': text_layer,
                'object_layer': object_layer,
                'background_layer': background_layer
            }
    
    def _refine_layer_alpha(self, layers: Dict, target: np.ndarray, composed: np.ndarray):
        """Refine alpha channels of layers based on composition error."""
        try:
            # Calculate error map
            error = cv2.absdiff(target, composed)
            error_gray = cv2.cvtColor(error, cv2.COLOR_BGR2GRAY)
            
            # Adjust text layer alpha where error is high
            text_mask = layers['text_layer'][:, :, 3] > 0
            high_error_text = np.logical_and(text_mask, error_gray > 20)
            
            if np.any(high_error_text):
                # Reduce alpha slightly in high error regions
                layers['text_layer'][:, :, 3][high_error_text] = np.clip(
                    layers['text_layer'][:, :, 3][high_error_text] * 0.9, 0, 255
                ).astype(np.uint8)
            
            # Similar for object layer
            object_mask = layers['object_layer'][:, :, 3] > 0
            high_error_object = np.logical_and(object_mask, error_gray > 20)
            
            if np.any(high_error_object):
                layers['object_layer'][:, :, 3][high_error_object] = np.clip(
                    layers['object_layer'][:, :, 3][high_error_object] * 0.9, 0, 255
                ).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Alpha refinement failed: {e}")
    
    def calculate_quality_metrics(self, original: np.ndarray, composed: np.ndarray) -> Dict:
        """Calculate comprehensive quality metrics."""
        try:
            # Ensure same dimensions
            if original.shape != composed.shape:
                composed = cv2.resize(composed, (original.shape[1], original.shape[0]))
            
            # Convert to float for calculations
            orig_float = original.astype(np.float64)
            comp_float = composed.astype(np.float64)
            
            # Mean Squared Error
            mse_value = mse(orig_float, comp_float)
            
            # Peak Signal-to-Noise Ratio
            psnr_value = psnr(original, composed, data_range=255)
            
            # Structural Similarity Index
            # Convert to grayscale for SSIM
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            comp_gray = cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY)
            ssim_value = ssim(orig_gray, comp_gray, data_range=255)
            
            return {
                'mse': float(mse_value),
                'psnr': float(psnr_value),
                'ssim': float(ssim_value)
            }
            
        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e}")
            return {'mse': float('inf'), 'psnr': 0.0, 'ssim': 0.0}
    
    def blend_layers(self, base: np.ndarray, overlay: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """Blend layers using different blending modes."""
        try:
            # Extract alpha if overlay has it
            if overlay.shape[2] == 4:
                alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                overlay_rgb = overlay[:, :, :3]
            else:
                alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.float32)
                overlay_rgb = overlay
            
            base_float = base.astype(np.float32)
            overlay_float = overlay_rgb.astype(np.float32)
            
            if mode == 'normal':
                result = overlay_float * alpha + base_float * (1 - alpha)
            elif mode == 'multiply':
                blended = (base_float * overlay_float) / 255.0
                result = blended * alpha + base_float * (1 - alpha)
            elif mode == 'screen':
                blended = 255 - ((255 - base_float) * (255 - overlay_float)) / 255.0
                result = blended * alpha + base_float * (1 - alpha)
            else:
                # Default to normal blending
                result = overlay_float * alpha + base_float * (1 - alpha)
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Layer blending failed: {e}")
            return base.copy()