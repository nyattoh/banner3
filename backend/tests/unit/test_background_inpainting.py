import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.background_inpainting import BackgroundInpaintingService


class TestBackgroundInpaintingService:
    """Test cases for background inpainting functionality."""
    
    @pytest.fixture
    def inpainting_service(self):
        """Create a BackgroundInpaintingService instance for testing."""
        return BackgroundInpaintingService()
    
    def test_init_background_inpainting_service(self, inpainting_service):
        """Test that BackgroundInpaintingService initializes correctly."""
        assert inpainting_service is not None
        assert hasattr(inpainting_service, 'inpaint_background')
        assert hasattr(inpainting_service, 'generate_background_layer')
    
    def test_create_combined_mask(self, inpainting_service):
        """Test creation of combined mask from text and object masks."""
        # Create sample masks
        text_mask = np.zeros((100, 100), dtype=np.uint8)
        text_mask[20:40, 20:60] = 255  # Text region
        
        object_mask = np.zeros((100, 100), dtype=np.uint8)
        object_mask[50:80, 30:70] = 255  # Object region
        
        combined_mask = inpainting_service.create_combined_mask(text_mask, object_mask)
        
        assert isinstance(combined_mask, np.ndarray)
        assert combined_mask.shape == (100, 100)
        assert combined_mask.dtype == np.uint8
        
        # Check that both regions are included
        assert np.any(combined_mask[20:40, 20:60] == 255)  # Text region
        assert np.any(combined_mask[50:80, 30:70] == 255)  # Object region
    
    def test_inpaint_background_opencv_ns(self, inpainting_service, sample_banner_image):
        """Test OpenCV Navier-Stokes inpainting."""
        # Create a mask to inpaint
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[100:200, 150:300] = 255  # Region to inpaint
        
        result = inpainting_service.inpaint_background_opencv(
            sample_banner_image, mask, method='ns'
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
        assert result.dtype == sample_banner_image.dtype
        
        # The inpainted region should be different from the original
        original_region = sample_banner_image[100:200, 150:300]
        inpainted_region = result[100:200, 150:300]
        assert not np.array_equal(original_region, inpainted_region)
    
    def test_inpaint_background_opencv_telea(self, inpainting_service, sample_banner_image):
        """Test OpenCV Telea inpainting."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[50:150, 100:250] = 255
        
        result = inpainting_service.inpaint_background_opencv(
            sample_banner_image, mask, method='telea'
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
        assert result.dtype == sample_banner_image.dtype
    
    def test_inpaint_background_with_empty_mask(self, inpainting_service, sample_banner_image):
        """Test inpainting with empty mask (no regions to inpaint)."""
        empty_mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        
        result = inpainting_service.inpaint_background_opencv(sample_banner_image, empty_mask)
        
        # Result should be identical to original when mask is empty
        assert np.array_equal(result, sample_banner_image)
    
    def test_inpaint_background_with_full_mask(self, inpainting_service, sample_banner_image):
        """Test inpainting with mask covering entire image."""
        full_mask = np.ones(sample_banner_image.shape[:2], dtype=np.uint8) * 255
        
        result = inpainting_service.inpaint_background_opencv(sample_banner_image, full_mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
        # Should handle edge case gracefully
    
    def test_edge_preserving_inpainting(self, inpainting_service, sample_banner_image):
        """Test edge-preserving inpainting method."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[80:120, 200:280] = 255
        
        result = inpainting_service.inpaint_edge_preserving(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
    
    def test_texture_synthesis_inpainting(self, inpainting_service, sample_banner_image):
        """Test texture synthesis inpainting."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[60:140, 180:260] = 255
        
        result = inpainting_service.inpaint_texture_synthesis(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
    
    def test_generate_background_layer(self, inpainting_service, sample_banner_image):
        """Test complete background layer generation."""
        # Create sample masks
        text_mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        text_mask[300:350, 300:500] = 255  # Text region
        
        object_mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        object_mask[150:250, 150:250] = 255  # Object region
        
        background_layer = inpainting_service.generate_background_layer(
            sample_banner_image, text_mask, object_mask
        )
        
        assert isinstance(background_layer, np.ndarray)
        assert background_layer.shape == sample_banner_image.shape
        
        # Background should be different in masked regions
        assert not np.array_equal(
            background_layer[300:350, 300:500],
            sample_banner_image[300:350, 300:500]
        )
    
    def test_inpainting_quality_assessment(self, inpainting_service):
        """Test assessment of inpainting quality."""
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        inpainted = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        quality_score = inpainting_service.assess_inpainting_quality(
            original, inpainted, mask
        )
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_inpaint_background_hybrid(self, inpainting_service, sample_banner_image):
        """Test hybrid inpainting method combining multiple techniques."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[100:200, 200:350] = 255
        
        result = inpainting_service.inpaint_background_hybrid(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
    
    def test_mask_dilation_erosion(self, inpainting_service):
        """Test mask preprocessing with dilation and erosion."""
        # Create a small mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255
        
        # Test dilation
        dilated = inpainting_service.dilate_mask(mask, kernel_size=3)
        assert np.sum(dilated) > np.sum(mask)  # Should be larger
        
        # Test erosion
        eroded = inpainting_service.erode_mask(mask, kernel_size=3)
        assert np.sum(eroded) < np.sum(mask)  # Should be smaller
    
    def test_adaptive_inpainting_radius(self, inpainting_service):
        """Test adaptive inpainting radius based on mask size."""
        small_mask = np.zeros((100, 100), dtype=np.uint8)
        small_mask[45:55, 45:55] = 255
        
        large_mask = np.zeros((100, 100), dtype=np.uint8)
        large_mask[20:80, 20:80] = 255
        
        small_radius = inpainting_service.calculate_adaptive_radius(small_mask)
        large_radius = inpainting_service.calculate_adaptive_radius(large_mask)
        
        assert isinstance(small_radius, int)
        assert isinstance(large_radius, int)
        assert large_radius > small_radius  # Larger mask should get larger radius
    
    def test_inpainting_with_invalid_mask(self, inpainting_service, sample_banner_image):
        """Test inpainting with invalid mask dimensions."""
        invalid_mask = np.zeros((50, 50), dtype=np.uint8)  # Wrong size
        
        with pytest.raises(ValueError):
            inpainting_service.inpaint_background_opencv(sample_banner_image, invalid_mask)
    
    def test_inpainting_performance(self, inpainting_service, sample_banner_image):
        """Test that inpainting completes within reasonable time."""
        import time
        
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[100:300, 200:400] = 255  # Large region to inpaint
        
        start_time = time.time()
        result = inpainting_service.inpaint_background_opencv(sample_banner_image, mask)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 15.0  # Should complete within 15 seconds
        assert isinstance(result, np.ndarray)
    
    def test_seamless_cloning_inpainting(self, inpainting_service, sample_banner_image):
        """Test seamless cloning for inpainting."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[120:180, 250:320] = 255
        
        result = inpainting_service.inpaint_seamless_cloning(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
    
    def test_progressive_inpainting(self, inpainting_service, sample_banner_image):
        """Test progressive inpainting for large areas."""
        # Create a large mask area
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[50:350, 100:500] = 255  # Large area
        
        result = inpainting_service.inpaint_progressive(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
    
    def test_background_pattern_detection(self, inpainting_service, sample_banner_image):
        """Test detection of background patterns for better inpainting."""
        pattern_info = inpainting_service.detect_background_patterns(sample_banner_image)
        
        assert isinstance(pattern_info, dict)
        assert 'dominant_colors' in pattern_info
        assert 'texture_features' in pattern_info
    
    def test_context_aware_inpainting(self, inpainting_service, sample_banner_image):
        """Test context-aware inpainting that considers surrounding regions."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[150:250, 200:300] = 255
        
        result = inpainting_service.inpaint_context_aware(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape
        
        # Context-aware method should produce better results
        # This would need more sophisticated quality metrics in practice
    
    def test_multi_scale_inpainting(self, inpainting_service, sample_banner_image):
        """Test multi-scale inpainting approach."""
        mask = np.zeros(sample_banner_image.shape[:2], dtype=np.uint8)
        mask[80:220, 180:320] = 255
        
        result = inpainting_service.inpaint_multiscale(sample_banner_image, mask)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_banner_image.shape