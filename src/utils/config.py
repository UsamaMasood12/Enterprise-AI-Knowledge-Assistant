"""
Configuration Management for Enterprise AI Assistant
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application
    APP_NAME: str = "Enterprise AI Assistant"
    LOG_LEVEL: str = "INFO"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: Optional[str] = None
    
    # OpenAI (will use later)
    OPENAI_API_KEY: Optional[str] = None
    
    # Pinecone (will use later)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS_PER_DOC: int = 500
    
    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 10
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        for directory in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.EMBEDDINGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Create global settings instance
settings = Settings()

# Supported file types
SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'word': ['.doc', '.docx'],
    'excel': ['.xls', '.xlsx', '.csv'],
    'text': ['.txt', '.md', '.rst'],
    'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
    'web': ['.html', '.htm']
}

# All supported extensions
ALL_EXTENSIONS = [ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts]
