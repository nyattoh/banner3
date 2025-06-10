from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import uuid
import aiofiles
import os
from datetime import datetime
import logging
from PIL import Image
import io

from app.core.config import settings
from app.models.schemas import ImageUploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demo (in production, use database)
uploaded_images = {}


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image file for processing.
    
    - **file**: Image file (PNG, JPG, JPEG)
    - Returns upload information with unique ID
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Check file size
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size // (1024*1024)}MB"
            )
        
        # Validate image format by trying to open it
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or corrupted data"
            )
        
        # Generate unique ID
        image_id = str(uuid.uuid4())
        
        # Save file
        file_path = settings.upload_dir / f"{image_id}{file_ext}"
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Store metadata
        upload_info = {
            "id": image_id,
            "filename": file.filename,
            "file_size": file_size,
            "file_path": str(file_path),
            "upload_time": datetime.utcnow(),
            "status": "uploaded",
            "content_type": file.content_type
        }
        
        uploaded_images[image_id] = upload_info
        
        logger.info(f"Image uploaded successfully: {image_id} ({file.filename})")
        
        return ImageUploadResponse(
            id=image_id,
            filename=file.filename,
            file_size=file_size,
            upload_time=upload_info["upload_time"],
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during upload"
        )


@router.get("/images")
async def list_images():
    """
    List all uploaded images.
    
    Returns a list of uploaded image metadata.
    """
    try:
        images = []
        for image_id, info in uploaded_images.items():
            images.append({
                "id": image_id,
                "filename": info["filename"],
                "file_size": info["file_size"],
                "upload_time": info["upload_time"].isoformat(),
                "status": info["status"]
            })
        
        # Sort by upload time (newest first)
        images.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return images
        
    except Exception as e:
        logger.error(f"Failed to list images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve image list"
        )


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """
    Delete an uploaded image and its processed results.
    
    - **image_id**: Unique identifier of the image to delete
    """
    try:
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        image_info = uploaded_images[image_id]
        
        # Delete original file
        if os.path.exists(image_info["file_path"]):
            os.remove(image_info["file_path"])
        
        # Delete processed files if they exist
        base_path = settings.static_dir / image_id
        if os.path.exists(base_path):
            import shutil
            shutil.rmtree(base_path)
        
        # Remove from memory
        del uploaded_images[image_id]
        
        logger.info(f"Image deleted successfully: {image_id}")
        
        return {"message": "Image deleted successfully", "id": image_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete image"
        )


@router.get("/images/{image_id}")
async def get_image_info(image_id: str):
    """
    Get information about a specific uploaded image.
    
    - **image_id**: Unique identifier of the image
    """
    try:
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        info = uploaded_images[image_id]
        
        return {
            "id": image_id,
            "filename": info["filename"],
            "file_size": info["file_size"],
            "upload_time": info["upload_time"].isoformat(),
            "status": info["status"],
            "content_type": info["content_type"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image info {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve image information"
        )