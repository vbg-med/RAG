import logging

from llama_index.embeddings.ollama import OllamaEmbedding
from src.config import Config

logger = logging.getLogger(__name__)


class Embedder:
    """Configures the embedding model used for indexing and retrieval"""

    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self._initialize_embedder()

    def _initialize_embedder(self):
        try:
            logger.info(
                f"Initializing Ollama Embedding with model "
                f"'{self.config.OLLAMA_EMBED_MODEL}' "
                f"at {self.config.OLLAMA_BASE_URL}..."
            )

            self.embed_model = OllamaEmbedding(
                model_name=self.config.OLLAMA_EMBED_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
            )

            logger.info(
                f"Successfully initialized Ollama embeddings with model "
                f"{self.config.OLLAMA_EMBED_MODEL}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama embedding: {e}"
            )
            raise

    def get_embedding_model(self):
        return self.embed_model