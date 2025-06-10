from fastapi import APIRouter, HTTPException, BackgroundTasks
import asyncio
import logging
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from app.core.config import settings
from app.models.schemas import ProcessingStatus, ProcessingResult
from app.services.text_detection import TextDetectionService
from app.services.object_detection import ObjectDetectionService
from app.services.background_inpainting import BackgroundInpaintingService
from app.services.layer_composition import LayerCompositionService
from app.api.upload import uploaded_images

router = APIRouter()
logger = logging.getLogger(__name__)

# Processing status storage
processing_status = {}
processing_results = {}

# Initialize services
text_service = TextDetectionService()
object_service = ObjectDetectionService()
background_service = BackgroundInpaintingService()
composition_service = LayerCompositionService()


@router.post("/process/{image_id}")
async def start_processing(image_id: str, background_tasks: BackgroundTasks):
    """
    Start processing an uploaded image to decompose it into layers.
    
    - **image_id**: Unique identifier of the uploaded image
    """
    try:
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Check if already processing
        if image_id in processing_status:
            current_status = processing_status[image_id]
            if current_status["status"] == "processing":
                raise HTTPException(
                    status_code=409,
                    detail="Image is already being processed"
                )
        
        # Initialize processing status
        processing_status[image_id] = {
            "id": image_id,
            "status": "processing",
            "progress": 0,
            "message": "Starting processing...",
            "error": None,
            "started_at": datetime.utcnow()
        }
        
        # Start background processing
        background_tasks.add_task(process_image_background, image_id)
        
        logger.info(f"Started processing image: {image_id}")
        
        return ProcessingStatus(
            id=image_id,
            status="processing",
            progress=0,
            message="Processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start processing"
        )


async def process_image_background(image_id: str):
    """Background task to process image and generate layers."""
    try:
        # Update status
        update_processing_status(image_id, 5, "Loading image...")
        
        # Load image
        image_info = uploaded_images[image_id]
        image_path = image_info["file_path"]
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")
        
        original_image = image.copy()
        
        # Create output directory
        output_dir = settings.static_dir / image_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Text Detection
        update_processing_status(image_id, 20, "Detecting text...")
        text_result = text_service.detect_text(image)
        text_mask = text_service.generate_text_mask(image, text_result)
        text_layer = text_service.extract_text_layer(image, text_result)
        
        # Save text layer
        text_layer_path = output_dir / "text_layer.png"
        cv2.imwrite(str(text_layer_path), text_layer)
        
        # Step 2: Object Detection
        update_processing_status(image_id, 40, "Detecting objects...")
        object_result = object_service.detect_objects(image)
        object_mask = object_service.generate_object_mask(image, object_result)
        object_layer = object_service.extract_object_layer(image, object_result)
        
        # Save object layer
        object_layer_path = output_dir / "object_layer.png"
        cv2.imwrite(str(object_layer_path), object_layer)
        
        # Step 3: Background Generation
        update_processing_status(image_id, 60, "Generating background...")
        background_layer = background_service.generate_background_layer(
            image, text_mask, object_mask
        )
        
        # Save background layer
        background_layer_path = output_dir / "background_layer.png"
        cv2.imwrite(str(background_layer_path), background_layer)
        
        # Step 4: Validation
        update_processing_status(image_id, 80, "Validating composition...")
        
        # Compose layers back together
        composed = composition_service.compose_layers(
            background_layer, text_layer, object_layer
        )
        
        # Validate composition
        validation = composition_service.validate_composition(original_image, composed)
        
        # Save composed image for comparison
        composed_path = output_dir / "composed.png"
        cv2.imwrite(str(composed_path), composed)
        
        # Step 5: Calculate metrics
        update_processing_status(image_id, 90, "Calculating quality metrics...")
        quality_metrics = composition_service.calculate_quality_metrics(
            original_image, composed
        )
        
        # Create result
        processing_time = (datetime.utcnow() - processing_status[image_id]["started_at"]).total_seconds()
        
        result = ProcessingResult(
            id=image_id,
            original_image=f"/static/{image_id}/original.png",
            layers=[
                {
                    "layer_type": "text",
                    "filename": "text_layer.png",
                    "file_size": os.path.getsize(text_layer_path),
                    "dimensions": (image.shape[1], image.shape[0])
                },
                {
                    "layer_type": "object",
                    "filename": "object_layer.png", 
                    "file_size": os.path.getsize(object_layer_path),
                    "dimensions": (image.shape[1], image.shape[0])
                },
                {
                    "layer_type": "background",
                    "filename": "background_layer.png",
                    "file_size": os.path.getsize(background_layer_path),
                    "dimensions": (image.shape[1], image.shape[0])
                }
            ],
            processing_time=processing_time,
            completed_at=datetime.utcnow(),
            download_urls={
                "text": f"/api/download/{image_id}/text",
                "object": f"/api/download/{image_id}/object",
                "background": f"/api/download/{image_id}/background",
                "all": f"/api/download/{image_id}/all"
            }
        )
        
        # Save original image to static directory for comparison
        original_path = output_dir / "original.png"
        cv2.imwrite(str(original_path), original_image)
        
        # Store results
        processing_results[image_id] = {
            "result": result,
            "validation": validation,
            "quality_metrics": quality_metrics,
            "text_detection": {
                "regions": len(text_result.text_regions),
                "detected_text": text_result.detected_text
            },
            "object_detection": {
                "regions": len(object_result.object_regions),
                "detected_classes": object_result.object_classes
            }
        }
        
        # Update final status
        update_processing_status(image_id, 100, "Processing completed successfully")
        processing_status[image_id]["status"] = "completed"
        
        # Update image status in uploads
        uploaded_images[image_id]["status"] = "processed"
        
        logger.info(f"Processing completed for image: {image_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for image {image_id}: {str(e)}")
        
        # Update error status
        processing_status[image_id].update({
            "status": "failed",
            "progress": 0,
            "message": "Processing failed",
            "error": str(e)
        })


def update_processing_status(image_id: str, progress: int, message: str):
    """Update processing status."""
    if image_id in processing_status:
        processing_status[image_id].update({
            "progress": progress,
            "message": message,
            "updated_at": datetime.utcnow()
        })


@router.get("/process/{image_id}/cancel")
async def cancel_processing(image_id: str):
    """
    Cancel ongoing processing for an image.
    
    - **image_id**: Unique identifier of the image
    """
    try:
        if image_id not in processing_status:
            raise HTTPException(
                status_code=404,
                detail="No processing found for this image"
            )
        
        current_status = processing_status[image_id]
        
        if current_status["status"] != "processing":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel processing in status: {current_status['status']}"
            )
        
        # Update status to cancelled
        processing_status[image_id].update({
            "status": "cancelled",
            "message": "Processing cancelled by user",
            "cancelled_at": datetime.utcnow()
        })
        
        logger.info(f"Processing cancelled for image: {image_id}")
        
        return {"message": "Processing cancelled", "id": image_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel processing {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel processing"
        )