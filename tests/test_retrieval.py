import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock llama_index and third-party modules if they are not installed in the current environment
try:
    import dotenv
    import pydantic
    import llama_index
except ImportError:

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
        "chromadb",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

try:
    import llama_index.core.settings

    llama_index.core.settings.resolve_embed_model = lambda x: x
    llama_index.core.settings.resolve_llm = lambda x: x
except Exception:
    pass

from src.config import Config


class TestRetrieval(unittest.TestCase):
    @patch("src.retrieval.retriever.chromadb.PersistentClient")
    @patch("src.retrieval.retriever.ChromaVectorStore")
    @patch("src.retrieval.retriever.Embedder")
    @patch("src.retrieval.retriever.ServiceContext")
    def test_retriever_initialization(
        self, mock_sc, mock_embedder, mock_cvs, mock_chroma
    ):
        config = Config()
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client

        from src.retrieval.retriever import Retriever, HAS_SETTINGS

        retriever = Retriever(config)

        # Verify chroma client paths and parameters
        mock_chroma.assert_called_once_with(path=str(config.CHROMA_DB_PATH))
        mock_client.get_or_create_collection.assert_called_once_with("rag_pipeline")

        if HAS_SETTINGS:
            try:
                from llama_index.core import Settings
            except ImportError:
                from llama_index.settings import Settings
            self.assertIsNotNone(Settings.embed_model)
        else:
            self.assertIsNotNone(retriever.service_context)

    @patch("src.retrieval.retriever.chromadb.PersistentClient")
    @patch("src.retrieval.retriever.ChromaVectorStore")
    @patch("src.retrieval.retriever.Embedder")
    @patch("src.retrieval.retriever.ServiceContext")
    @patch("src.retrieval.retriever.VectorStoreIndex")
    @patch("src.retrieval.retriever.StorageContext")
    def test_build_index(
        self, mock_storage_ctx, mock_vsi, mock_sc, mock_embedder, mock_cvs, mock_chroma
    ):
        config = Config()
        from src.retrieval.retriever import Retriever

        retriever = Retriever(config)

        mock_chunks = [MagicMock()]
        retriever.build_index(mock_chunks)

        # Verify LlamaIndex StorageContext and VectorStoreIndex were created
        mock_storage_ctx.from_defaults.assert_called_once_with(
            vector_store=retriever.vector_store
        )
        mock_vsi.assert_called_once()

    @patch("src.retrieval.retriever.chromadb.PersistentClient")
    @patch("src.retrieval.retriever.ChromaVectorStore")
    @patch("src.retrieval.retriever.Embedder")
    @patch("src.retrieval.retriever.ServiceContext")
    def test_retrieve_filtering(self, mock_sc, mock_embedder, mock_cvs, mock_chroma):
        config = Config()
        from src.retrieval.retriever import Retriever

        retriever = Retriever(config)

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        # Set up nodes with scores: nws1 is above threshold (0.5), nws2 is below
        node1 = MagicMock(text="Content 1")
        node2 = MagicMock(text="Content 2")
        nws1 = MagicMock(score=0.8, node=node1)
        nws2 = MagicMock(score=0.3, node=node2)

        mock_retriever.retrieve.return_value = [nws1, nws2]

        results = retriever.retrieve(mock_index, "test query", top_k=2, threshold=0.5)

        # Only node1 passes the score threshold of 0.5
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], node1)

    @patch("src.retrieval.retriever.chromadb.PersistentClient")
    @patch("src.retrieval.retriever.ChromaVectorStore")
    @patch("src.retrieval.retriever.Embedder")
    @patch("src.retrieval.retriever.ServiceContext")
    def test_retrieve_safety_fallback(
        self, mock_sc, mock_embedder, mock_cvs, mock_chroma
    ):
        config = Config()
        from src.retrieval.retriever import Retriever

        retriever = Retriever(config)

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever

        # Both nodes are below threshold (0.5)
        node1 = MagicMock(text="Content 1")
        node2 = MagicMock(text="Content 2")
        nws1 = MagicMock(score=0.4, node=node1)
        nws2 = MagicMock(score=0.2, node=node2)

        mock_retriever.retrieve.return_value = [nws1, nws2]

        results = retriever.retrieve(mock_index, "test query", top_k=2, threshold=0.5)

        # Since all nodes are below threshold, safety fallback should return the top match (node1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], node1)


if __name__ == "__main__":
    unittest.main()
