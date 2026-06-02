import logging
import ollama
from src.config import Config

logger = logging.getLogger(__name__)


class OllamaEmbeddingWrapper:
    """Wrapper for direct Ollama embedding API to work with LlamaIndex"""

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url
        # Set the Ollama host for the ollama package
        ollama.api.client = ollama.Client(host=base_url)

    def get_text_embedding(self, text: str) -> list:
        """Get embedding for a single text using ollama.embed()"""
        try:
            response = ollama.embed(
                model=self.model_name,
                input=text
            )
            return response['embeddings'][0] if response['embeddings'] else []
        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise

    def get_text_embeddings(self, texts: list) -> list:
        """Get embeddings for multiple texts"""
        try:
            response = ollama.embed(
                model=self.model_name,
                input=texts
            )
            return response['embeddings']
        except Exception as e:
            logger.error(f"Error getting embeddings for texts: {e}")
            raise

    def embed_batch(self, texts: list) -> list:
        """Batch embedding for compatibility"""
        return self.get_text_embeddings(texts)


class Embedder:
    """Configures the embedding model used for indexing and query retrieval"""

    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self._initialize_embedder()

    def _initialize_embedder(self):
        try:
            logger.info(
                f"Initializing Ollama Embedding with model '{self.config.OLLAMA_EMBED_MODEL}' "
                f"at {self.config.OLLAMA_BASE_URL}..."
            )
            self.embed_model = OllamaEmbeddingWrapper(
                model_name=self.config.OLLAMA_EMBED_MODEL,
                base_url=self.config.OLLAMA_BASE_URL,
            )
            logger.info(f"Successfully initialized Ollama embeddings with model {self.config.OLLAMA_EMBED_MODEL}")
        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama embedding: {e}"
            )
            raise

    def get_embedding_model(self):
        """Return the configured embedding model instance"""
        return self.embed_model
