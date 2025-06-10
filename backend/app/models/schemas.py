from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import datetime
import uuid


class ImageUploadResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    upload_time: datetime
    status: str = "uploaded"


class ProcessingStatus(BaseModel):
    id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100
    message: str
    error: Optional[str] = None


class LayerInfo(BaseModel):
    layer_type: str  # "text", "background", "object"
    filename: str
    file_size: int
    dimensions: Tuple[int, int]


class ProcessingResult(BaseModel):
    id: str
    original_image: str
    layers: List[LayerInfo]
    processing_time: float
    completed_at: datetime
    download_urls: dict


class TextDetectionResult(BaseModel):
    text_regions: List[List[int]]  # Bounding boxes [x, y, w, h]
    confidence_scores: List[float]
    detected_text: List[str]


class ObjectDetectionResult(BaseModel):
    object_regions: List[List[int]]  # Bounding boxes [x, y, w, h]
    confidence_scores: List[float]
    object_classes: List[str]


class LayerCompositionValidation(BaseModel):
    is_valid: bool
    similarity_score: float
    errors: List[str]