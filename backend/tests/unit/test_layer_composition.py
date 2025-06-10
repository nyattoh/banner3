import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.layer_composition import LayerCompositionService
from app.models.schemas import LayerCompositionValidation


class TestLayerCompositionService:
    """Test cases for layer composition and validation functionality."""
    
    @pytest.fixture
    def composition_service(self):
        """Create a LayerCompositionService instance for testing."""
        return LayerCompositionService()
    
    @pytest.fixture
    def sample_layers(self, sample_banner_image):
        """Create sample layer images for testing."""
        height, width = sample_banner_image.shape[:2]
        
        # Text layer (RGBA)
        text_layer = np.zeros((height, width, 4), dtype=np.uint8)
        text_layer[50:100, 300:500, :3] = [255, 0, 0]  # Red text
        text_layer[50:100, 300:500, 3] = 255  # Opaque
        
        # Object layer (RGBA)
        object_layer = np.zeros((height, width, 4), dtype=np.uint8)
        object_layer[150:250, 150:250, :3] = [0, 0, 255]  # Blue object
        object_layer[150:250, 150:250, 3] = 255  # Opaque
        
        # Background layer (RGB)
        background_layer = sample_banner_image.copy()
        
        return {
            'text_layer': text_layer,
            'object_layer': object_layer,
            'background_layer': background_layer,
            'original': sample_banner_image
        }
    
    def test_init_layer_composition_service(self, composition_service):
        """Test that LayerCompositionService initializes correctly."""
        assert composition_service is not None
        assert hasattr(composition_service, 'compose_layers')
        assert hasattr(composition_service, 'validate_composition')
    
    def test_compose_layers_basic(self, composition_service, sample_layers):
        """Test basic layer composition."""
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=sample_layers['text_layer'],
            object_layer=sample_layers['object_layer']
        )
        
        assert isinstance(composed, np.ndarray)
        assert composed.shape == sample_layers['original'].shape
        assert composed.dtype == np.uint8
        
        # Check that text and object are present in composition
        # Text region should be red
        text_region = composed[50:100, 300:500]
        assert np.any(text_region[:, :, 0] > 200)  # Red channel
        
        # Object region should be blue
        object_region = composed[150:250, 150:250]
        assert np.any(object_region[:, :, 2] > 200)  # Blue channel
    
    def test_compose_layers_alpha_blending(self, composition_service, sample_layers):
        """Test alpha blending in layer composition."""
        # Create semi-transparent text layer
        text_layer = sample_layers['text_layer'].copy()
        text_layer[:, :, 3] = 128  # 50% transparent
        
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=text_layer,
            object_layer=sample_layers['object_layer']
        )
        
        assert isinstance(composed, np.ndarray)
        # Semi-transparent text should blend with background
        # The result should not be pure red or pure background color
    
    def test_compose_layers_empty_text(self, composition_service, sample_layers):
        """Test composition with empty text layer."""
        empty_text = np.zeros_like(sample_layers['text_layer'])
        
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=empty_text,
            object_layer=sample_layers['object_layer']
        )
        
        assert isinstance(composed, np.ndarray)
        # Should still show object layer and background
    
    def test_compose_layers_empty_object(self, composition_service, sample_layers):
        """Test composition with empty object layer."""
        empty_object = np.zeros_like(sample_layers['object_layer'])
        
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=sample_layers['text_layer'],
            object_layer=empty_object
        )
        
        assert isinstance(composed, np.ndarray)
        # Should still show text layer and background
    
    def test_compose_layers_overlapping(self, composition_service, sample_layers):
        """Test composition with overlapping text and object layers."""
        # Move text layer to overlap with object
        text_layer = sample_layers['text_layer'].copy()
        text_layer[150:200, 200:300, :3] = [255, 255, 0]  # Yellow text
        text_layer[150:200, 200:300, 3] = 255
        
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=text_layer,
            object_layer=sample_layers['object_layer']
        )
        
        assert isinstance(composed, np.ndarray)
        # Overlapping region should show text on top of object
        overlap_region = composed[150:200, 200:250]
        assert np.any(overlap_region[:, :, 0] > 200)  # Red channel from yellow text
        assert np.any(overlap_region[:, :, 1] > 200)  # Green channel from yellow text
    
    def test_validate_composition_perfect_match(self, composition_service, sample_layers):
        """Test validation when composition perfectly matches original."""
        original = sample_layers['original']
        
        # Create a perfect composition (same as original)
        composed = original.copy()
        
        validation = composition_service.validate_composition(original, composed)
        
        assert isinstance(validation, LayerCompositionValidation)
        assert validation.is_valid is True
        assert validation.similarity_score >= 0.95  # Very high similarity
        assert len(validation.errors) == 0
    
    def test_validate_composition_poor_match(self, composition_service, sample_layers):
        """Test validation when composition poorly matches original."""
        original = sample_layers['original']
        
        # Create a poor composition (random noise)
        poor_composed = np.random.randint(0, 255, original.shape, dtype=np.uint8)
        
        validation = composition_service.validate_composition(original, poor_composed)
        
        assert isinstance(validation, LayerCompositionValidation)
        assert validation.is_valid is False
        assert validation.similarity_score < 0.5  # Low similarity
        assert len(validation.errors) > 0
    
    def test_calculate_similarity_identical(self, composition_service):
        """Test similarity calculation for identical images."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        similarity = composition_service.calculate_similarity(image, image)
        
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self, composition_service):
        """Test similarity calculation for different images."""
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        similarity = composition_service.calculate_similarity(image1, image2)
        
        assert similarity < 0.5  # Very different images
    
    def test_detect_composition_errors(self, composition_service, sample_layers):
        """Test detection of specific composition errors."""
        original = sample_layers['original']
        
        # Create composition with artifacts
        composed = original.copy()
        composed[0:50, 0:50] = [255, 0, 255]  # Magenta artifact
        
        errors = composition_service.detect_composition_errors(original, composed)
        
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any('artifact' in error.lower() or 'color' in error.lower() for error in errors)
    
    def test_check_layer_completeness(self, composition_service, sample_layers):
        """Test checking if layers cover the entire original image."""
        completeness = composition_service.check_layer_completeness(
            original=sample_layers['original'],
            text_layer=sample_layers['text_layer'],
            object_layer=sample_layers['object_layer'],
            background_layer=sample_layers['background_layer']
        )
        
        assert isinstance(completeness, dict)
        assert 'coverage_percentage' in completeness
        assert 'missing_regions' in completeness
        assert 0.0 <= completeness['coverage_percentage'] <= 100.0
    
    def test_blend_layers_with_masks(self, composition_service):
        """Test layer blending using custom masks."""
        # Create simple test layers
        layer1 = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White
        layer2 = np.zeros((50, 50, 3), dtype=np.uint8)  # Black
        
        # Create blend mask (gradient)
        mask = np.linspace(0, 255, 50).astype(np.uint8)
        mask = np.tile(mask, (50, 1))
        
        blended = composition_service.blend_layers_with_mask(layer1, layer2, mask)
        
        assert isinstance(blended, np.ndarray)
        assert blended.shape == layer1.shape
        # Should have gradient from white to black
    
    def test_extract_layer_differences(self, composition_service, sample_layers):
        """Test extraction of differences between original and composed."""
        original = sample_layers['original']
        composed = original.copy()
        composed[100:150, 100:150] = [255, 0, 0]  # Add red patch
        
        diff_map = composition_service.extract_layer_differences(original, composed)
        
        assert isinstance(diff_map, np.ndarray)
        assert diff_map.shape[:2] == original.shape[:2]
        # Difference should be highlighted in the modified region
        assert np.any(diff_map[100:150, 100:150] > 0)
    
    def test_optimize_layer_composition(self, composition_service, sample_layers):
        """Test optimization of layer composition."""
        optimized_layers = composition_service.optimize_layer_composition(
            text_layer=sample_layers['text_layer'],
            object_layer=sample_layers['object_layer'],
            background_layer=sample_layers['background_layer'],
            target=sample_layers['original']
        )
        
        assert isinstance(optimized_layers, dict)
        assert 'text_layer' in optimized_layers
        assert 'object_layer' in optimized_layers
        assert 'background_layer' in optimized_layers
    
    def test_handle_layer_transparency_edge_cases(self, composition_service):
        """Test handling of edge cases in layer transparency."""
        # Create layers with various transparency patterns
        base_layer = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Fully transparent layer
        transparent_layer = np.zeros((100, 100, 4), dtype=np.uint8)
        
        # Fully opaque layer
        opaque_layer = np.ones((100, 100, 4), dtype=np.uint8) * 255
        opaque_layer[:, :, 3] = 255
        
        result1 = composition_service.compose_layers(base_layer, transparent_layer)
        result2 = composition_service.compose_layers(base_layer, opaque_layer)
        
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        
        # Transparent layer should not change background
        assert np.array_equal(result1, base_layer)
        
        # Opaque layer should completely replace background
        assert np.array_equal(result2[:, :, :3], opaque_layer[:, :, :3])
    
    def test_composition_performance(self, composition_service, sample_layers):
        """Test that layer composition completes within reasonable time."""
        import time
        
        start_time = time.time()
        
        # Compose layers multiple times
        for _ in range(5):
            composed = composition_service.compose_layers(
                background=sample_layers['background_layer'],
                text_layer=sample_layers['text_layer'],
                object_layer=sample_layers['object_layer']
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 5.0  # Should complete 5 compositions in under 5 seconds
        assert isinstance(composed, np.ndarray)
    
    def test_color_space_consistency(self, composition_service, sample_layers):
        """Test that color space is maintained throughout composition."""
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=sample_layers['text_layer'],
            object_layer=sample_layers['object_layer']
        )
        
        # Check color space consistency
        assert composed.dtype == sample_layers['background_layer'].dtype
        assert composed.shape[2] == 3  # RGB output
        assert np.all(composed >= 0) and np.all(composed <= 255)
    
    def test_validate_layer_dimensions(self, composition_service):
        """Test validation of layer dimension compatibility."""
        # Create layers with different dimensions
        background = np.zeros((100, 100, 3), dtype=np.uint8)
        wrong_size_text = np.zeros((50, 50, 4), dtype=np.uint8)
        correct_size_object = np.zeros((100, 100, 4), dtype=np.uint8)
        
        # Should raise error for mismatched dimensions
        with pytest.raises(ValueError):
            composition_service.compose_layers(background, wrong_size_text, correct_size_object)
    
    def test_composition_quality_metrics(self, composition_service, sample_layers):
        """Test calculation of composition quality metrics."""
        composed = composition_service.compose_layers(
            background=sample_layers['background_layer'],
            text_layer=sample_layers['text_layer'],
            object_layer=sample_layers['object_layer']
        )
        
        metrics = composition_service.calculate_quality_metrics(
            original=sample_layers['original'],
            composed=composed
        )
        
        assert isinstance(metrics, dict)
        assert 'mse' in metrics  # Mean Squared Error
        assert 'psnr' in metrics  # Peak Signal-to-Noise Ratio
        assert 'ssim' in metrics  # Structural Similarity Index
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_adaptive_blending_modes(self, composition_service):
        """Test different blending modes for layer composition."""
        base = np.ones((50, 50, 3), dtype=np.uint8) * 128
        overlay = np.ones((50, 50, 4), dtype=np.uint8) * 255
        overlay[:, :, 3] = 128  # Semi-transparent
        
        # Test different blending modes
        normal = composition_service.blend_layers(base, overlay, mode='normal')
        multiply = composition_service.blend_layers(base, overlay, mode='multiply')
        screen = composition_service.blend_layers(base, overlay, mode='screen')
        
        assert isinstance(normal, np.ndarray)
        assert isinstance(multiply, np.ndarray)
        assert isinstance(screen, np.ndarray)
        
        # Results should be different for different blend modes
        assert not np.array_equal(normal, multiply)
        assert not np.array_equal(normal, screen)