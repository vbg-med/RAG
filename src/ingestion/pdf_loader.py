from llama_index.readers.file import PDFReader
from src.config import Config
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFLoader:
    def __init__(self, config: Config):
        self.config = config
        self.reader = PDFReader()

    def load(self, pdf_path: str):
        """Load PDF and return documents"""
        try:
            logger.info(f"Loading {pdf_path}...")
            # Ensure path format is correct (Path object or path string)
            path_obj = Path(pdf_path)
            documents = self.reader.load_data(path_obj)

            # Enrich metadata
            for doc in documents:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = path_obj.name
                if "type" not in doc.metadata:
                    doc.metadata["type"] = "text"

            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return []
