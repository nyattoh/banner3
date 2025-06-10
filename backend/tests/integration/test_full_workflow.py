import pytest
import asyncio
import tempfile
import os
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
import io
import time


class TestFullWorkflow:
    """Integration tests for the complete image processing workflow."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for integration tests."""
        from app.main import app
        return TestClient(app)
    
    @pytest.fixture
    def sample_banner_image(self):
        """Create a realistic banner image for testing."""
        # Create a banner-like image with text and objects
        img = Image.new('RGB', (800, 400), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Add background gradient
        for y in range(400):
            color_value = int(200 - (y / 400) * 50)
            draw.line([(0, y), (800, y)], fill=(color_value, color_value + 20, color_value + 40))
        
        # Add a product-like rectangle (simulating an object)
        draw.rectangle([150, 100, 300, 250], fill='white', outline='black', width=3)
        draw.rectangle([160, 110, 290, 240], fill='red')
        
        # Add another object
        draw.ellipse([500, 150, 650, 300], fill='yellow', outline='orange', width=2)
        
        # Add text elements
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 36)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Main heading
        draw.text((50, 50), "SUPER SALE", fill='darkblue', font=font_large)
        
        # Subheading
        draw.text((400, 100), "50% OFF", fill='red', font=font_large)
        
        # Additional text
        draw.text((50, 320), "Limited Time Offer", fill='black', font=font_medium)
        draw.text((400, 350), "Shop Now!", fill='darkgreen', font=font_medium)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    
    def test_complete_workflow_success(self, client, sample_banner_image):
        """Test the complete workflow from upload to download."""
        
        # Step 1: Upload image
        files = {"file": ("test_banner.png", sample_banner_image, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        image_id = upload_data["id"]
        
        assert "id" in upload_data
        assert upload_data["filename"] == "test_banner.png"
        assert upload_data["status"] == "uploaded"
        
        # Step 2: Start processing
        process_response = client.post(f"/api/process/{image_id}")
        
        assert process_response.status_code == 200
        process_data = process_response.json()
        assert process_data["status"] == "processing"
        assert process_data["id"] == image_id
        
        # Step 3: Monitor processing status
        max_wait_time = 120  # 2 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = client.get(f"/api/status/{image_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Processing failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(2)  # Wait 2 seconds before checking again
        else:
            pytest.fail("Processing did not complete within the time limit")
        
        # Step 4: Get results
        results_response = client.get(f"/api/results/{image_id}")
        assert results_response.status_code == 200
        
        results_data = results_response.json()
        assert "processing_result" in results_data
        assert "validation" in results_data
        assert "quality_metrics" in results_data
        
        # Verify processing result structure
        processing_result = results_data["processing_result"]
        assert len(processing_result["layers"]) == 3
        
        layer_types = [layer["layer_type"] for layer in processing_result["layers"]]
        assert "text" in layer_types
        assert "object" in layer_types
        assert "background" in layer_types
        
        # Verify validation results
        validation = results_data["validation"]
        assert "is_valid" in validation
        assert "similarity_score" in validation
        assert validation["similarity_score"] >= 0.0
        assert validation["similarity_score"] <= 1.0
        
        # Step 5: Download individual layers
        layer_types = ["text", "object", "background", "original", "composed"]
        
        for layer_type in layer_types:
            download_response = client.get(f"/api/download/{image_id}/{layer_type}")
            
            if download_response.status_code == 200:
                assert download_response.headers["content-type"] == "image/png"
                assert len(download_response.content) > 0
            else:
                # Some layers might not exist depending on processing results
                assert download_response.status_code in [404, 400]
        
        # Step 6: Download all layers as ZIP
        zip_response = client.get(f"/api/download/{image_id}/all")
        
        if zip_response.status_code == 200:
            assert zip_response.headers["content-type"] == "application/zip"
            assert len(zip_response.content) > 0
        
        # Step 7: Clean up - delete the image
        delete_response = client.delete(f"/api/images/{image_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        status_after_delete = client.get(f"/api/status/{image_id}")
        assert status_after_delete.status_code == 404
    
    def test_workflow_with_invalid_image(self, client):
        """Test workflow with invalid image data."""
        
        # Upload invalid data
        invalid_data = b"This is not an image"
        files = {"file": ("invalid.png", io.BytesIO(invalid_data), "image/png")}
        
        upload_response = client.post("/api/upload", files=files)
        assert upload_response.status_code == 400
        
        error_data = upload_response.json()
        assert "detail" in error_data
        assert "invalid" in error_data["detail"].lower()
    
    def test_workflow_with_large_file(self, client):
        """Test workflow with file exceeding size limit."""
        
        # Create a large image (over 10MB)
        large_img = Image.new('RGB', (5000, 5000), color='white')
        img_bytes = io.BytesIO()
        large_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"file": ("large.png", img_bytes, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        
        # Should be rejected due to size
        assert upload_response.status_code == 413
    
    def test_process_nonexistent_image(self, client):
        """Test processing with non-existent image ID."""
        
        fake_id = "non-existent-id"
        process_response = client.post(f"/api/process/{fake_id}")
        
        assert process_response.status_code == 404
        assert "not found" in process_response.json()["detail"].lower()
    
    def test_download_before_processing(self, client, sample_banner_image):
        """Test attempting to download layers before processing is complete."""
        
        # Upload image
        files = {"file": ("test.png", sample_banner_image, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Try to download without processing
        download_response = client.get(f"/api/download/{image_id}/text")
        assert download_response.status_code in [400, 404]
    
    def test_concurrent_uploads(self, client):
        """Test handling multiple concurrent uploads."""
        
        def create_test_image(color):
            img = Image.new('RGB', (200, 200), color=color)
            draw = ImageDraw.Draw(img)
            draw.text((50, 50), f"Test {color}", fill='black')
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            return img_bytes
        
        # Upload multiple images
        colors = ['red', 'green', 'blue']
        uploaded_ids = []
        
        for color in colors:
            test_image = create_test_image(color)
            files = {"file": (f"test_{color}.png", test_image, "image/png")}
            
            response = client.post("/api/upload", files=files)
            assert response.status_code == 200
            
            uploaded_ids.append(response.json()["id"])
        
        # Verify all uploads were successful
        assert len(uploaded_ids) == 3
        assert len(set(uploaded_ids)) == 3  # All IDs should be unique
        
        # Clean up
        for image_id in uploaded_ids:
            client.delete(f"/api/images/{image_id}")
    
    def test_api_health_check(self, client):
        """Test API health check endpoint."""
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_api_documentation_accessible(self, client):
        """Test that API documentation is accessible."""
        
        # Test OpenAPI schema
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        
        schema = openapi_response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Test Swagger UI
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        assert "text/html" in docs_response.headers["content-type"]
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        
        # Preflight request
        options_response = client.options("/api/upload", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        })
        
        # Check CORS headers
        assert "access-control-allow-origin" in options_response.headers
    
    def test_processing_status_transitions(self, client, sample_banner_image):
        """Test that processing status transitions correctly."""
        
        # Upload image
        files = {"file": ("status_test.png", sample_banner_image, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Initial status should be pending
        initial_status = client.get(f"/api/status/{image_id}")
        assert initial_status.status_code == 200
        assert initial_status.json()["status"] == "pending"
        
        # Start processing
        client.post(f"/api/process/{image_id}")
        
        # Status should change to processing
        processing_status = client.get(f"/api/status/{image_id}")
        assert processing_status.status_code == 200
        
        status_data = processing_status.json()
        assert status_data["status"] in ["processing", "completed"]
        
        if status_data["status"] == "processing":
            assert "progress" in status_data
            assert 0 <= status_data["progress"] <= 100
        
        # Clean up
        client.delete(f"/api/images/{image_id}")
    
    def test_error_handling_robustness(self, client):
        """Test error handling in various scenarios."""
        
        # Test with missing image ID
        empty_process = client.post("/api/process/")
        assert empty_process.status_code == 404
        
        # Test with malformed image ID
        malformed_status = client.get("/api/status/invalid-uuid-format")
        assert malformed_status.status_code == 404
        
        # Test download with invalid layer type
        fake_id = "test-id"
        invalid_layer = client.get(f"/api/download/{fake_id}/invalid_layer")
        assert invalid_layer.status_code in [400, 404]
    
    def test_file_cleanup_after_deletion(self, client, sample_banner_image):
        """Test that files are properly cleaned up after deletion."""
        
        # Upload and process image
        files = {"file": ("cleanup_test.png", sample_banner_image, "image/png")}
        upload_response = client.post("/api/upload", files=files)
        image_id = upload_response.json()["id"]
        
        # Delete the image
        delete_response = client.delete(f"/api/images/{image_id}")
        assert delete_response.status_code == 200
        
        # Verify that subsequent operations fail
        status_response = client.get(f"/api/status/{image_id}")
        assert status_response.status_code == 404
        
        download_response = client.get(f"/api/download/{image_id}/text")
        assert download_response.status_code == 404