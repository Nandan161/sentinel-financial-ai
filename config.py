import os
from dataclasses import dataclass

@dataclass
class Config:
    # File handling
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: list = (".pdf",)
    DATA_DIR: str = "data/raw"
    
    # Chunking
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    RETRIEVAL_K: int = 5
    
    # Model
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0
    
    # Vector store
    CHROMA_DIR: str = "chroma_db"
    
    # Caching
    CACHE_ENABLED: bool = True
    
    # Logging
    LOG_FILE: str = "sentinel_financial.log"
    LOG_LEVEL: str = "INFO"
    
    @classmethod
    def from_env(cls):
        """Load config from environment variables"""
        return cls(
            MAX_FILE_SIZE_MB=int(os.getenv("MAX_FILE_SIZE_MB", 100)),
            # ... other env vars
        )

config = Config()