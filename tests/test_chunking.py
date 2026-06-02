import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock llama_index and third-party modules if they are not installed in the current environment
# This allows tests to run successfully and verify control flow even without installations.
try:
    import dotenv
    import pydantic
    import llama_index
except ImportError:
    # Create fake modules in sys.modules
    class StubBaseModel:
        pass

    mock_pydantic = MagicMock()
    mock_pydantic.BaseModel = StubBaseModel
    sys.modules["pydantic"] = mock_pydantic

    for mod_name in [
        "dotenv",
        "llama_index",
        "llama_index.text_splitter",
        "llama_index.readers",
        "llama_index.readers.file",
        "llama_index.embeddings",
        "llama_index.embeddings.ollama",
        "llama_index.embeddings.huggingface",
        "llama_index.vector_stores",
        "llama_index.vector_stores.chroma",
        "llama_index.core",
        "llama_index.schema",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

from src.config import Config
from src.chunking.chunker import Chunker


class TestChunking(unittest.TestCase):
    @patch("src.chunking.chunker.SentenceSplitter")
    def test_chunker_initialization(self, mock_splitter_cls):
        config = Config()
        mock_splitter = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        chunker = Chunker(config)

        # Verify initialization arguments match configuration parameters
        mock_splitter_cls.assert_called_once_with(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separator="\n\n",
        )

    @patch("src.chunking.chunker.SentenceSplitter")
    def test_chunk_documents(self, mock_splitter_cls):
        config = Config()
        mock_splitter = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        # Setup mock return nodes for splitter
        mock_nodes = ["chunk1", "chunk2"]
        mock_splitter.get_nodes_from_documents.return_value = mock_nodes

        chunker = Chunker(config)
        mock_docs = [MagicMock(text="Some long document content")]

        result = chunker.chunk_documents(mock_docs)

        # Verify standard get_nodes_from_documents was executed
        mock_splitter.get_nodes_from_documents.assert_called_once_with(mock_docs)
        self.assertEqual(result, mock_nodes)

    @patch("src.chunking.chunker.SentenceSplitter")
    def test_chunk_documents_fallback(self, mock_splitter_cls):
        config = Config()
        # Delete get_nodes_from_documents to force fallback to split_documents
        mock_splitter = MagicMock(spec=["split_documents"])
        mock_splitter_cls.return_value = mock_splitter

        mock_nodes = ["chunk_fallback_1", "chunk_fallback_2"]
        mock_splitter.split_documents.return_value = mock_nodes

        chunker = Chunker(config)
        mock_docs = [MagicMock(text="Some long document content")]

        result = chunker.chunk_documents(mock_docs)

        # Verify fallback split_documents was called
        mock_splitter.split_documents.assert_called_once_with(mock_docs)
        self.assertEqual(result, mock_nodes)


if __name__ == "__main__":
    unittest.main()
