from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import logging
import os
import zipfile
import io
from datetime import datetime

from app.core.config import settings
from app.api.upload import uploaded_images
from app.api.process import processing_status, processing_results

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/download/{image_id}/{layer_type}")
async def download_layer(image_id: str, layer_type: str):
    """
    Download a specific layer of a processed image.
    
    - **image_id**: Unique identifier of the image
    - **layer_type**: Type of layer to download (text, object, background, original, composed)
    """
    try:
        # Validate inputs
        valid_layer_types = ["text", "object", "background", "original", "composed"]
        if layer_type not in valid_layer_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer type. Must be one of: {', '.join(valid_layer_types)}"
            )
        
        # Check if image exists
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Check if processing is complete
        if image_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail="Processing not started for this image"
            )
        
        status_info = processing_status[image_id]
        if status_info["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Processing not completed. Current status: {status_info['status']}"
            )
        
        # Construct file path
        output_dir = settings.static_dir / image_id
        
        if layer_type == "text":
            file_path = output_dir / "text_layer.png"
            filename = f"{image_id}_text_layer.png"
        elif layer_type == "object":
            file_path = output_dir / "object_layer.png"
            filename = f"{image_id}_object_layer.png"
        elif layer_type == "background":
            file_path = output_dir / "background_layer.png"
            filename = f"{image_id}_background_layer.png"
        elif layer_type == "original":
            file_path = output_dir / "original.png"
            filename = f"{image_id}_original.png"
        elif layer_type == "composed":
            file_path = output_dir / "composed.png"
            filename = f"{image_id}_composed.png"
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Layer file not found: {layer_type}"
            )
        
        logger.info(f"Downloading {layer_type} layer for image: {image_id}")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Layer-Type": layer_type,
                "X-Image-ID": image_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download {layer_type} layer for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download layer"
        )


@router.get("/download/{image_id}/all")
async def download_all_layers(image_id: str):
    """
    Download all layers of a processed image as a ZIP file.
    
    - **image_id**: Unique identifier of the image
    """
    try:
        # Check if image exists
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Check if processing is complete
        if image_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail="Processing not started for this image"
            )
        
        status_info = processing_status[image_id]
        if status_info["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Processing not completed. Current status: {status_info['status']}"
            )
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            output_dir = settings.static_dir / image_id
            
            # Define files to include
            files_to_zip = [
                ("original.png", f"{image_id}_original.png"),
                ("text_layer.png", f"{image_id}_text_layer.png"),
                ("object_layer.png", f"{image_id}_object_layer.png"),
                ("background_layer.png", f"{image_id}_background_layer.png"),
                ("composed.png", f"{image_id}_composed.png")
            ]
            
            # Add files to ZIP
            for source_filename, zip_filename in files_to_zip:
                file_path = output_dir / source_filename
                if os.path.exists(file_path):
                    zip_file.write(file_path, zip_filename)
                else:
                    logger.warning(f"File not found for ZIP: {file_path}")
            
            # Add metadata JSON
            if image_id in processing_results:
                metadata = {
                    "image_id": image_id,
                    "processing_date": datetime.utcnow().isoformat(),
                    "original_filename": uploaded_images[image_id]["filename"],
                    "validation": processing_results[image_id]["validation"].dict(),
                    "quality_metrics": processing_results[image_id]["quality_metrics"],
                    "detection_summary": {
                        "text_detection": processing_results[image_id]["text_detection"],
                        "object_detection": processing_results[image_id]["object_detection"]
                    }
                }
                
                metadata_json = io.BytesIO(
                    str.encode(str(metadata).replace("'", '"'))
                )
                zip_file.writestr(f"{image_id}_metadata.json", metadata_json.getvalue())
        
        zip_buffer.seek(0)
        
        filename = f"{image_id}_layers.zip"
        
        logger.info(f"Downloading all layers as ZIP for image: {image_id}")
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Image-ID": image_id,
                "X-Archive-Type": "all-layers"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download all layers for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create ZIP archive"
        )


@router.get("/preview/{image_id}/{layer_type}")
async def preview_layer(image_id: str, layer_type: str):
    """
    Preview a specific layer (returns image for display, not download).
    
    - **image_id**: Unique identifier of the image
    - **layer_type**: Type of layer to preview
    """
    try:
        # Validate inputs
        valid_layer_types = ["text", "object", "background", "original", "composed"]
        if layer_type not in valid_layer_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer type. Must be one of: {', '.join(valid_layer_types)}"
            )
        
        # Check if image exists
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # For original image, allow preview even if not processed
        if layer_type == "original":
            original_path = uploaded_images[image_id]["file_path"]
            if os.path.exists(original_path):
                return FileResponse(
                    path=original_path,
                    media_type="image/png",
                    headers={"X-Layer-Type": layer_type}
                )
        
        # For other layers, require processing to be complete
        if image_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail="Processing not started for this image"
            )
        
        status_info = processing_status[image_id]
        if status_info["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Processing not completed. Current status: {status_info['status']}"
            )
        
        # Construct file path
        output_dir = settings.static_dir / image_id
        
        layer_files = {
            "text": "text_layer.png",
            "object": "object_layer.png", 
            "background": "background_layer.png",
            "original": "original.png",
            "composed": "composed.png"
        }
        
        file_path = output_dir / layer_files[layer_type]
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Layer file not found: {layer_type}"
            )
        
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            headers={
                "X-Layer-Type": layer_type,
                "X-Image-ID": image_id,
                "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview {layer_type} layer for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to preview layer"
        )


@router.get("/download/{image_id}/comparison")
async def download_comparison_image(image_id: str):
    """
    Download a side-by-side comparison of original and composed images.
    
    - **image_id**: Unique identifier of the image
    """
    try:
        # Check if processing is complete
        if image_id not in processing_status or processing_status[image_id]["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail="Processing not completed"
            )
        
        output_dir = settings.static_dir / image_id
        original_path = output_dir / "original.png"
        composed_path = output_dir / "composed.png"
        
        if not (os.path.exists(original_path) and os.path.exists(composed_path)):
            raise HTTPException(
                status_code=404,
                detail="Comparison images not found"
            )
        
        # Create side-by-side comparison using OpenCV
        import cv2
        import numpy as np
        
        original = cv2.imread(str(original_path))
        composed = cv2.imread(str(composed_path))
        
        # Resize if needed to ensure same dimensions
        if original.shape != composed.shape:
            composed = cv2.resize(composed, (original.shape[1], original.shape[0]))
        
        # Create side-by-side comparison
        comparison = np.hstack([original, composed])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Composed", (original.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        # Save comparison image
        comparison_path = output_dir / "comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        
        filename = f"{image_id}_comparison.png"
        
        return FileResponse(
            path=str(comparison_path),
            filename=filename,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create comparison for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create comparison image"
        )