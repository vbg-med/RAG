try:
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    try:
        from llama_index.text_splitter import SentenceSplitter
    except ImportError:
        from llama_index.node_parser import SentenceSplitter
from src.config import Config
import logging

logger = logging.getLogger(__name__)


class Chunker:
    """Chunks Document objects into smaller text blocks (nodes) for retrieval and context construction"""

    def __init__(self, config: Config):
        self.config = config
        self.splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separator="\n\n",
        )

    def chunk_documents(self, documents):
        """Split documents into chunks"""
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks...")

            # Robust compatibility check across LlamaIndex versions
            if hasattr(self.splitter, "get_nodes_from_documents"):
                chunks = self.splitter.get_nodes_from_documents(documents)
            elif hasattr(self.splitter, "split_documents"):
                chunks = self.splitter.split_documents(documents)
            else:
                chunks = self.splitter(documents)

            logger.info(f"Generated {len(chunks)} chunks from documents.")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return []
