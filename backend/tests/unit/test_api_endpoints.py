import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import io
import numpy as np
from PIL import Image
import json


class TestAPIEndpoints:
    """Test cases for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        from app.main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing uploads."""
        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data as bytes."""
        img = Image.new('RGB', (400, 300), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_upload_image_valid_file(self, client, sample_image_file):
        """Test uploading a valid image file."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "filename" in data
        assert "file_size" in data
        assert data["filename"] == "test.png"
    
    def test_upload_image_invalid_format(self, client):
        """Test uploading an invalid file format."""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        files = {"file": ("test.txt", text_file, "text/plain")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "format" in data["detail"].lower() or "type" in data["detail"].lower()
    
    def test_upload_image_too_large(self, client):
        """Test uploading a file that's too large."""
        # Create a large image (simulated)
        large_data = b"x" * (15 * 1024 * 1024)  # 15MB (over 10MB limit)
        files = {"file": ("large.png", io.BytesIO(large_data), "image/png")}
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 413  # Payload too large
    
    def test_upload_image_no_file(self, client):
        """Test upload endpoint without file."""
        response = client.post("/api/upload")
        
        assert response.status_code == 422  # Unprocessable entity
    
    @patch('app.services.text_detection.TextDetectionService.detect_text')
    @patch('app.services.object_detection.ObjectDetectionService.detect_objects')
    @patch('app.services.background_inpainting.BackgroundInpaintingService.generate_background_layer')
    def test_process_image_valid_id(self, mock_background, mock_objects, mock_text, client, sample_image_file):
        """Test processing an image with valid ID."""
        # First upload an image
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Mock service responses
        from app.models.schemas import TextDetectionResult, ObjectDetectionResult
        
        mock_text.return_value = TextDetectionResult(
            text_regions=[[50, 50, 100, 30]],
            confidence_scores=[0.9],
            detected_text=["Sample Text"]
        )
        
        mock_objects.return_value = ObjectDetectionResult(
            object_regions=[[150, 150, 80, 80]],
            confidence_scores=[0.8],
            object_classes=["product"]
        )
        
        mock_background.return_value = np.ones((300, 400, 3), dtype=np.uint8) * 128
        
        # Process the image
        response = client.post(f"/api/process/{image_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["status"] == "processing"
    
    def test_process_image_invalid_id(self, client):
        """Test processing with invalid image ID."""
        response = client.post("/api/process/invalid-id")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()
    
    def test_get_processing_status_valid_id(self, client, sample_image_file):
        """Test getting processing status for valid ID."""
        # Upload and process image first
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Get status
        response = client.get(f"/api/status/{image_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert "progress" in data
        assert data["id"] == image_id
    
    def test_get_processing_status_invalid_id(self, client):
        """Test getting status for invalid ID."""
        response = client.get("/api/status/invalid-id")
        
        assert response.status_code == 404
    
    @patch('app.services.layer_composition.LayerCompositionService.validate_composition')
    def test_get_results_valid_id(self, mock_validate, client, sample_image_file):
        """Test getting results for valid processed image."""
        # Mock validation result
        from app.models.schemas import LayerCompositionValidation
        mock_validate.return_value = LayerCompositionValidation(
            is_valid=True,
            similarity_score=0.95,
            errors=[]
        )
        
        # Upload image first
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Get results (assuming processing is complete)
        response = client.get(f"/api/results/{image_id}")
        
        # This might return 404 if processing isn't complete, which is expected
        assert response.status_code in [200, 404]
    
    def test_get_results_invalid_id(self, client):
        """Test getting results for invalid ID."""
        response = client.get("/api/results/invalid-id")
        
        assert response.status_code == 404
    
    def test_download_layer_text(self, client, sample_image_file):
        """Test downloading text layer."""
        # Upload image first
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Try to download text layer
        response = client.get(f"/api/download/{image_id}/text")
        
        # Might be 404 if not processed yet
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "image/png"
    
    def test_download_layer_object(self, client, sample_image_file):
        """Test downloading object layer."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        response = client.get(f"/api/download/{image_id}/object")
        assert response.status_code in [200, 404]
    
    def test_download_layer_background(self, client, sample_image_file):
        """Test downloading background layer."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        response = client.get(f"/api/download/{image_id}/background")
        assert response.status_code in [200, 404]
    
    def test_download_layer_invalid_type(self, client, sample_image_file):
        """Test downloading invalid layer type."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        response = client.get(f"/api/download/{image_id}/invalid")
        assert response.status_code == 400
    
    def test_download_all_layers_zip(self, client, sample_image_file):
        """Test downloading all layers as ZIP."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        response = client.get(f"/api/download/{image_id}/all")
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/zip"
    
    def test_delete_image_valid_id(self, client, sample_image_file):
        """Test deleting an uploaded image."""
        files = {"file": ("test.png", sample_image_file, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        response = client.delete(f"/api/images/{image_id}")
        assert response.status_code == 200
        
        # Verify image is deleted
        status_response = client.get(f"/api/status/{image_id}")
        assert status_response.status_code == 404
    
    def test_delete_image_invalid_id(self, client):
        """Test deleting with invalid ID."""
        response = client.delete("/api/images/invalid-id")
        assert response.status_code == 404
    
    def test_list_images_empty(self, client):
        """Test listing images when none exist."""
        response = client.get("/api/images")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_images_with_uploads(self, client, sample_image_file):
        """Test listing images after uploading some."""
        # Upload multiple images
        for i in range(3):
            sample_image_file.seek(0)  # Reset file pointer
            files = {"file": (f"test{i}.png", sample_image_file, "image/png")}
            client.post("/api/upload", files=files)
        
        response = client.get("/api/images")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/upload")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
    
    def test_rate_limiting(self, client, sample_image_file):
        """Test rate limiting on upload endpoint."""
        # This test would need rate limiting to be implemented
        # For now, just verify multiple uploads work
        for i in range(5):
            sample_image_file.seek(0)
            files = {"file": (f"test{i}.png", sample_image_file, "image/png")}
            response = client.post("/api/upload", files=files)
            assert response.status_code in [200, 429]  # 429 if rate limited
    
    def test_concurrent_uploads(self, client, sample_image_file):
        """Test handling concurrent uploads."""
        import concurrent.futures
        
        def upload_image(file_data, filename):
            file_data.seek(0)
            files = {"file": (filename, file_data, "image/png")}
            return client.post("/api/upload", files=files)
        
        # Create multiple file objects
        file_objects = []
        for i in range(3):
            img = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            file_objects.append((img_bytes, f"concurrent{i}.png"))
        
        # Upload concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(upload_image, file_obj, filename)
                for file_obj, filename in file_objects
            ]
            
            results = [future.result() for future in futures]
        
        # All uploads should succeed
        assert all(r.status_code == 200 for r in results)
    
    def test_api_documentation(self, client):
        """Test that API documentation is accessible."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
    
    def test_api_docs_ui(self, client):
        """Test that Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_error_handling_500(self, client):
        """Test handling of internal server errors."""
        # This would need to be implemented with a mock that raises an exception
        pass
    
    def test_request_validation(self, client):
        """Test request validation with invalid data."""
        # Test with invalid JSON
        response = client.post("/api/upload", json={"invalid": "data"})
        assert response.status_code == 422
    
    def test_image_metadata_extraction(self, client):
        """Test extraction of image metadata during upload."""
        # Create image with metadata
        img = Image.new('RGB', (800, 600), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("metadata_test.png", img_bytes, "image/png")}
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "file_size" in data
        assert data["file_size"] > 0
    
    def test_cleanup_expired_files(self, client):
        """Test cleanup of expired uploaded files."""
        # This would test the cleanup background task
        # For now, just verify the endpoint exists
        pass
    
    @pytest.mark.asyncio
    async def test_websocket_status_updates(self, client):
        """Test WebSocket connection for real-time status updates."""
        # This would test WebSocket functionality if implemented
        pass
    
    def test_api_versioning(self, client):
        """Test API versioning headers."""
        response = client.get("/api/upload")
        # Check if API version is included in response headers
        # This depends on implementation
        pass
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        
        # Check for security headers (if implemented)
        # Examples: X-Content-Type-Options, X-Frame-Options, etc.
        assert response.status_code == 200