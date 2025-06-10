from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # App settings
    app_name: str = "Banner Layer Decomposition API"
    debug: bool = True
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    upload_dir: Path = base_dir / "uploads"
    static_dir: Path = base_dir / "static"
    temp_dir: Path = base_dir / "temp"
    
    # File settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: set = {".jpg", ".jpeg", ".png"}
    
    # Processing settings
    tesseract_cmd: str = "/usr/bin/tesseract"
    yolo_model_path: str = "yolov8n.pt"
    
    # API settings
    cors_origins: list = ["http://localhost:3000", "http://frontend:3000"]
    
    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.static_dir, exist_ok=True)
os.makedirs(settings.temp_dir, exist_ok=True)