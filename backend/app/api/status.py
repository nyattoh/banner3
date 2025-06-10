from fastapi import APIRouter, HTTPException
import logging
from datetime import datetime

from app.models.schemas import ProcessingStatus, ProcessingResult
from app.api.process import processing_status, processing_results
from app.api.upload import uploaded_images

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/status/{image_id}", response_model=ProcessingStatus)
async def get_processing_status(image_id: str):
    """
    Get the current processing status of an image.
    
    - **image_id**: Unique identifier of the image
    - Returns current processing status and progress
    """
    try:
        # Check if image exists
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Check if processing has been started
        if image_id not in processing_status:
            # Image uploaded but not yet processed
            return ProcessingStatus(
                id=image_id,
                status="pending",
                progress=0,
                message="Image uploaded, processing not started"
            )
        
        status_info = processing_status[image_id]
        
        return ProcessingStatus(
            id=image_id,
            status=status_info["status"],
            progress=status_info["progress"],
            message=status_info["message"],
            error=status_info.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve processing status"
        )


@router.get("/results/{image_id}")
async def get_processing_results(image_id: str):
    """
    Get the processing results for a completed image.
    
    - **image_id**: Unique identifier of the image
    - Returns detailed results including layers and validation info
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
        
        # Get results
        if image_id not in processing_results:
            raise HTTPException(
                status_code=404,
                detail="Processing results not found"
            )
        
        results = processing_results[image_id]
        
        return {
            "processing_result": results["result"].dict(),
            "validation": results["validation"].dict(),
            "quality_metrics": results["quality_metrics"],
            "detection_summary": {
                "text_detection": results["text_detection"],
                "object_detection": results["object_detection"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get results for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve processing results"
        )


@router.get("/status")
async def get_all_processing_status():
    """
    Get processing status for all images.
    
    Returns a list of all processing statuses.
    """
    try:
        all_status = []
        
        # Include all uploaded images
        for image_id, image_info in uploaded_images.items():
            if image_id in processing_status:
                status_info = processing_status[image_id]
                all_status.append({
                    "id": image_id,
                    "filename": image_info["filename"],
                    "upload_time": image_info["upload_time"].isoformat(),
                    "status": status_info["status"],
                    "progress": status_info["progress"],
                    "message": status_info["message"],
                    "error": status_info.get("error")
                })
            else:
                # Not yet processed
                all_status.append({
                    "id": image_id,
                    "filename": image_info["filename"],
                    "upload_time": image_info["upload_time"].isoformat(),
                    "status": "pending",
                    "progress": 0,
                    "message": "Not yet processed",
                    "error": None
                })
        
        # Sort by upload time (newest first)
        all_status.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return all_status
        
    except Exception as e:
        logger.error(f"Failed to get all status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve status information"
        )


@router.get("/summary")
async def get_processing_summary():
    """
    Get a summary of processing statistics.
    
    Returns overall statistics about processed images.
    """
    try:
        total_images = len(uploaded_images)
        
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        total_processing_time = 0.0
        completed_count = 0
        
        for image_id in uploaded_images:
            if image_id in processing_status:
                status = processing_status[image_id]["status"]
                status_counts[status] += 1
                
                if status == "completed" and image_id in processing_results:
                    result = processing_results[image_id]["result"]
                    total_processing_time += result.processing_time
                    completed_count += 1
            else:
                status_counts["pending"] += 1
        
        average_processing_time = (
            total_processing_time / completed_count if completed_count > 0 else 0.0
        )
        
        return {
            "total_images": total_images,
            "status_breakdown": status_counts,
            "average_processing_time": round(average_processing_time, 2),
            "completed_images": completed_count,
            "success_rate": round(
                (completed_count / total_images * 100) if total_images > 0 else 0, 2
            )
        }
        
    except Exception as e:
        logger.error(f"Failed to get processing summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve processing summary"
        )


@router.delete("/status/{image_id}")
async def clear_processing_status(image_id: str):
    """
    Clear processing status and results for an image.
    
    - **image_id**: Unique identifier of the image
    """
    try:
        if image_id not in uploaded_images:
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Clear processing status
        if image_id in processing_status:
            del processing_status[image_id]
        
        # Clear results
        if image_id in processing_results:
            del processing_results[image_id]
        
        # Reset image status
        uploaded_images[image_id]["status"] = "uploaded"
        
        logger.info(f"Cleared processing status for image: {image_id}")
        
        return {"message": "Processing status cleared", "id": image_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear status for {image_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear processing status"
        )