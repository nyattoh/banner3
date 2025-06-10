import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.object_detection import ObjectDetectionService
from app.models.schemas import ObjectDetectionResult


class TestObjectDetectionService:
    """Test cases for object detection functionality."""
    
    @pytest.fixture
    def object_service(self):
        """Create an ObjectDetectionService instance for testing."""
        return ObjectDetectionService()
    
    def test_init_object_detection_service(self, object_service):
        """Test that ObjectDetectionService initializes correctly."""
        assert object_service is not None
        assert hasattr(object_service, 'detect_objects')
        assert hasattr(object_service, 'generate_object_mask')
    
    def test_detect_objects_with_valid_image(self, object_service, sample_banner_image):
        """Test object detection with a valid image containing objects."""
        result = object_service.detect_objects(sample_banner_image)
        
        assert isinstance(result, ObjectDetectionResult)
        assert len(result.object_regions) >= 0
        assert len(result.confidence_scores) == len(result.object_regions)
        assert len(result.object_classes) == len(result.object_regions)
        assert all(score >= 0 and score <= 1 for score in result.confidence_scores)
    
    def test_detect_objects_with_empty_image(self, object_service, sample_empty_image):
        """Test object detection with an empty image."""
        result = object_service.detect_objects(sample_empty_image)
        
        assert isinstance(result, ObjectDetectionResult)
        # Empty image should have no objects
        assert len(result.object_regions) == 0
        assert len(result.confidence_scores) == 0
        assert len(result.object_classes) == 0
    
    def test_generate_object_mask_basic(self, object_service, sample_banner_image):
        """Test object mask generation."""
        # First detect objects
        detection_result = object_service.detect_objects(sample_banner_image)
        
        # Generate mask
        mask = object_service.generate_object_mask(sample_banner_image, detection_result)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape[:2] == sample_banner_image.shape[:2]
        assert mask.dtype == np.uint8
        assert len(np.unique(mask)) <= 2  # Should be binary mask
    
    def test_generate_object_mask_with_empty_detection(self, object_service, sample_empty_image):
        """Test mask generation when no objects are detected."""
        empty_result = ObjectDetectionResult(
            object_regions=[],
            confidence_scores=[],
            object_classes=[]
        )
        
        mask = object_service.generate_object_mask(sample_empty_image, empty_result)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape[:2] == sample_empty_image.shape[:2]
        assert np.all(mask == 0)  # Should be all black (no objects)
    
    def test_object_confidence_threshold(self, object_service):
        """Test that low confidence detections are filtered out."""
        # Create mock detection with low confidence
        low_conf_result = ObjectDetectionResult(
            object_regions=[[10, 10, 50, 50], [100, 100, 30, 30]],
            confidence_scores=[0.1, 0.9],  # One low, one high confidence
            object_classes=["person", "car"]
        )
        
        filtered_result = object_service.filter_by_confidence(low_conf_result, threshold=0.5)
        
        assert len(filtered_result.object_regions) == 1
        assert filtered_result.confidence_scores[0] == 0.9
        assert filtered_result.object_classes[0] == "car"
    
    def test_object_region_validation(self, object_service):
        """Test that detected object regions are valid."""
        invalid_result = ObjectDetectionResult(
            object_regions=[[-1, -1, 10, 10], [0, 0, -5, 10]],  # Invalid coordinates
            confidence_scores=[0.9, 0.8],
            object_classes=["person", "car"]
        )
        
        with pytest.raises(ValueError):
            object_service.validate_object_regions(invalid_result)
    
    @patch('ultralytics.YOLO')
    def test_yolo_integration(self, mock_yolo, object_service, sample_banner_image):
        """Test integration with YOLO model."""
        # Mock YOLO model
        mock_model = Mock()
        mock_results = [Mock()]
        mock_results[0].boxes = Mock()
        mock_results[0].boxes.xyxy = np.array([[10, 10, 60, 60], [100, 100, 150, 150]])
        mock_results[0].boxes.conf = np.array([0.9, 0.8])
        mock_results[0].boxes.cls = np.array([0, 2])  # person, car classes
        mock_results[0].names = {0: 'person', 2: 'car'}
        
        mock_model.return_value = mock_results
        mock_yolo.return_value = mock_model
        
        result = object_service.detect_objects_yolo(sample_banner_image)
        
        assert isinstance(result, ObjectDetectionResult)
        assert len(result.object_regions) == 2
        mock_yolo.assert_called_once()
        mock_model.assert_called_once()
    
    def test_background_removal_integration(self, object_service, sample_banner_image):
        """Test integration with background removal (rembg)."""
        mask = object_service.generate_foreground_mask(sample_banner_image)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape[:2] == sample_banner_image.shape[:2]
        assert mask.dtype == np.uint8
    
    def test_object_detection_performance(self, object_service, sample_banner_image):
        """Test that object detection completes within reasonable time."""
        import time
        
        start_time = time.time()
        result = object_service.detect_objects(sample_banner_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert isinstance(result, ObjectDetectionResult)
    
    def test_multiple_object_types(self, object_service):
        """Test detection of various object types (person, product, etc.)."""
        # This would need sample images with different object types
        # Test that service can distinguish between different object classes
        pass
    
    def test_object_size_filtering(self, object_service):
        """Test filtering objects by size."""
        result = ObjectDetectionResult(
            object_regions=[[0, 0, 5, 5], [10, 10, 100, 100]],  # Small and large objects
            confidence_scores=[0.9, 0.9],
            object_classes=["small", "large"]
        )
        
        filtered = object_service.filter_by_size(result, min_area=50)
        
        assert len(filtered.object_regions) == 1
        assert filtered.object_regions[0] == [10, 10, 100, 100]
    
    def test_overlapping_objects_handling(self, object_service):
        """Test handling of overlapping object detections."""
        overlapping_result = ObjectDetectionResult(
            object_regions=[[10, 10, 50, 50], [20, 20, 50, 50]],  # Overlapping boxes
            confidence_scores=[0.9, 0.7],
            object_classes=["person", "person"]
        )
        
        # Should merge or choose best detection
        merged = object_service.handle_overlapping_objects(overlapping_result)
        
        assert len(merged.object_regions) <= 2
        assert isinstance(merged, ObjectDetectionResult)
    
    def test_object_classification_accuracy(self, object_service):
        """Test accuracy of object classification."""
        # Mock test with known object types
        # Verify that classifications are reasonable
        pass
    
    def test_extract_object_layer(self, object_service, sample_banner_image):
        """Test extraction of object layer as transparent PNG."""
        detection_result = object_service.detect_objects(sample_banner_image)
        object_layer = object_service.extract_object_layer(sample_banner_image, detection_result)
        
        assert isinstance(object_layer, np.ndarray)
        assert object_layer.shape[2] == 4  # RGBA
        assert object_layer.shape[:2] == sample_banner_image.shape[:2]
    
    def test_edge_case_tiny_objects(self, object_service):
        """Test handling of very small objects."""
        tiny_result = ObjectDetectionResult(
            object_regions=[[0, 0, 1, 1], [5, 5, 2, 2]],  # Very small objects
            confidence_scores=[0.9, 0.8],
            object_classes=["tiny1", "tiny2"]
        )
        
        # Service should handle tiny objects gracefully
        mask = object_service.generate_object_mask(np.zeros((100, 100, 3)), tiny_result)
        assert isinstance(mask, np.ndarray)
    
    def test_edge_case_large_objects(self, object_service):
        """Test handling of objects that span most of the image."""
        large_result = ObjectDetectionResult(
            object_regions=[[0, 0, 95, 95]],  # Almost entire image
            confidence_scores=[0.9],
            object_classes=["large_object"]
        )
        
        mask = object_service.generate_object_mask(np.zeros((100, 100, 3)), large_result)
        assert isinstance(mask, np.ndarray)
        assert np.sum(mask > 0) > 0  # Should detect the large object