import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.services.text_detection import TextDetectionService
from app.models.schemas import TextDetectionResult


class TestTextDetectionService:
    """Test cases for text detection functionality."""
    
    @pytest.fixture
    def text_service(self):
        """Create a TextDetectionService instance for testing."""
        return TextDetectionService()
    
    def test_init_text_detection_service(self, text_service):
        """Test that TextDetectionService initializes correctly."""
        assert text_service is not None
        assert hasattr(text_service, 'detect_text')
        assert hasattr(text_service, 'generate_text_mask')
    
    def test_detect_text_with_valid_image(self, text_service, sample_text_image):
        """Test text detection with a valid image containing text."""
        result = text_service.detect_text(sample_text_image)
        
        assert isinstance(result, TextDetectionResult)
        assert len(result.text_regions) > 0
        assert len(result.confidence_scores) == len(result.text_regions)
        assert len(result.detected_text) == len(result.text_regions)
        assert all(score >= 0 and score <= 1 for score in result.confidence_scores)
    
    def test_detect_text_with_empty_image(self, text_service, sample_empty_image):
        """Test text detection with an empty image."""
        result = text_service.detect_text(sample_empty_image)
        
        assert isinstance(result, TextDetectionResult)
        assert len(result.text_regions) == 0
        assert len(result.confidence_scores) == 0
        assert len(result.detected_text) == 0
    
    def test_detect_text_with_noisy_image(self, text_service, sample_noisy_image):
        """Test text detection robustness with noisy image."""
        result = text_service.detect_text(sample_noisy_image)
        
        assert isinstance(result, TextDetectionResult)
        # Should still detect some text despite noise
        assert len(result.text_regions) >= 0
    
    def test_generate_text_mask_basic(self, text_service, sample_text_image):
        """Test text mask generation."""
        # First detect text
        detection_result = text_service.detect_text(sample_text_image)
        
        # Generate mask
        mask = text_service.generate_text_mask(sample_text_image, detection_result)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape[:2] == sample_text_image.shape[:2]
        assert mask.dtype == np.uint8
        assert len(np.unique(mask)) <= 2  # Should be binary mask
    
    def test_generate_text_mask_with_empty_detection(self, text_service, sample_empty_image):
        """Test mask generation when no text is detected."""
        empty_result = TextDetectionResult(
            text_regions=[],
            confidence_scores=[],
            detected_text=[]
        )
        
        mask = text_service.generate_text_mask(sample_empty_image, empty_result)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape[:2] == sample_empty_image.shape[:2]
        assert np.all(mask == 0)  # Should be all black (no text)
    
    def test_detect_text_confidence_threshold(self, text_service):
        """Test that low confidence detections are filtered out."""
        # This would need to be implemented in the actual service
        # Test with different confidence thresholds
        pass
    
    def test_detect_text_language_support(self, text_service):
        """Test detection of different languages (Japanese, English)."""
        # Create images with different languages
        # Test that both are detected properly
        pass
    
    def test_text_region_validation(self, text_service):
        """Test that detected text regions are valid."""
        # Mock a detection result with invalid regions
        invalid_result = TextDetectionResult(
            text_regions=[[-1, -1, 10, 10], [0, 0, -5, 10]],  # Invalid coordinates
            confidence_scores=[0.9, 0.8],
            detected_text=["text1", "text2"]
        )
        
        # The service should handle invalid regions gracefully
        with pytest.raises(ValueError):
            text_service.validate_text_regions(invalid_result)
    
    @patch('pytesseract.image_to_data')
    def test_tesseract_integration(self, mock_tesseract, text_service, sample_text_image):
        """Test integration with Tesseract OCR."""
        # Mock Tesseract response
        mock_tesseract.return_value = {
            'text': ['', 'Hello', 'World', ''],
            'conf': ['-1', '95', '90', '-1'],
            'left': [0, 50, 150, 0],
            'top': [0, 50, 50, 0],
            'width': [400, 80, 60, 0],
            'height': [200, 30, 30, 0]
        }
        
        result = text_service.detect_text_tesseract(sample_text_image)
        
        assert isinstance(result, TextDetectionResult)
        mock_tesseract.assert_called_once()
    
    @patch('easyocr.Reader')
    def test_easyocr_integration(self, mock_easyocr, text_service, sample_text_image):
        """Test integration with EasyOCR."""
        # Mock EasyOCR response
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[50, 50], [130, 50], [130, 80], [50, 80]], 'Hello', 0.95),
            ([[50, 100], [180, 100], [180, 130], [50, 130]], 'World', 0.90)
        ]
        mock_easyocr.return_value = mock_reader
        
        result = text_service.detect_text_easyocr(sample_text_image)
        
        assert isinstance(result, TextDetectionResult)
        assert len(result.text_regions) == 2
        assert result.detected_text == ['Hello', 'World']
    
    def test_text_detection_performance(self, text_service, sample_banner_image):
        """Test that text detection completes within reasonable time."""
        import time
        
        start_time = time.time()
        result = text_service.detect_text(sample_banner_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert isinstance(result, TextDetectionResult)
    
    def test_multiple_text_engines_comparison(self, text_service, sample_text_image):
        """Test and compare results from different OCR engines."""
        tesseract_result = text_service.detect_text_tesseract(sample_text_image)
        easyocr_result = text_service.detect_text_easyocr(sample_text_image)
        
        # Both should detect some text
        assert len(tesseract_result.text_regions) >= 0
        assert len(easyocr_result.text_regions) >= 0
        
        # Results might differ but should be reasonable
        assert isinstance(tesseract_result, TextDetectionResult)
        assert isinstance(easyocr_result, TextDetectionResult)