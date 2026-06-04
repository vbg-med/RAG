"""Configuration management"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration"""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma_db"))
    LOG_FILE = os.getenv("LOG_FILE", BASE_DIR / "logs" / "pipeline.log")

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm:l6-v2")

    # LLM settings
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

    # RAG settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    # Image/Media processing
    ONLY_TEXT_EMBEDDING = os.getenv("ONLY_TEXT_EMBEDDING", "true").lower() == "true"
    USE_IMAGE_CAPTIONS = os.getenv("USE_IMAGE_CAPTIONS", "true").lower() == "true" and not ONLY_TEXT_EMBEDDING
    IMAGE_MODEL = os.getenv("IMAGE_MODEL", "llava")

    # Create directories
    @classmethod
    def create_dirs(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        (cls.DATA_DIR / "incoming").mkdir(exist_ok=True)
        (cls.DATA_DIR / "processed").mkdir(exist_ok=True)
        Path(cls.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


# Create dirs on import
Config.create_dirs()
