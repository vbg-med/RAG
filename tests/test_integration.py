import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub optional modules to prevent import failures
for mod_name in [
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
    "pypdf",
    "pdf2image",
    "PIL",
    "ollama",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

try:
    import llama_index.core.settings

    llama_index.core.settings.resolve_embed_model = lambda x: x
    llama_index.core.settings.resolve_llm = lambda x: x
except Exception:
    pass

# Check if fastapi package is installed in this python environment
try:
    from fastapi.testclient import TestClient
    from src.main import app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipIf(
    not FASTAPI_AVAILABLE,
    "FastAPI requirements not installed yet; skipping integration test.",
)
class TestIntegration(unittest.TestCase):
    """Integration test suite checking endpoint coordination and HTTP mapping"""

    def setUp(self):
        self.client = TestClient(app)

    @patch("src.main.retriever")
    @patch("src.main.llm_generator")
    def test_query_endpoint_success(self, mock_llm, mock_retriever):
        # Mock active global index state
        import src.main

        src.main.app_state.index = MagicMock()

        # Mock retrieval node returns
        mock_node = MagicMock()
        mock_node.text = (
            "In Q3 2023, revenue increased by 15% due to high cloud demand."
        )
        mock_node.metadata = {
            "source": "earnings_report.pdf",
            "page": "4",
            "type": "text",
        }
        mock_retriever.retrieve.return_value = [mock_node]

        # Mock LLM generation output
        mock_llm.generate.return_value = ("Revenue increased by 15% in Q3.", 0.88)

        response = self.client.post(
            "/query", json={"question": "What was the revenue increase in Q3?"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Revenue increased by 15% in Q3.")
        self.assertEqual(data["confidence"], 0.88)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["source"], "earnings_report.pdf")
        self.assertEqual(data["sources"][0]["page"], "4")

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("index_loaded", data)

    def test_query_without_index_raises_503(self):
        # Force null global index
        import src.main

        src.main.app_state.index = None

        response = self.client.post("/query", json={"question": "Where is the index?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("Index not initialized", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
