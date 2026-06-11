# RAG Pipeline - Local Document Retrieval & Generation System

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with Python, designed to process PDF documents, extract content with OCR, chunk and embed text, and generate answers using a local LLM (Ollama) with high-quality confidence scoring.

## Features

- **PDF Processing**: Extracts text and images from PDFs with OCR support
- **Document Chunking**: Smart text chunking with configurable overlap and chunk sizes
- **Vector Embeddings**: Uses HuggingFace embeddings stored in ChromaDB
- **Local LLM Integration**: Runs inference via Ollama for privacy and control
- **Confidence Scoring**: Weighted multi-component scoring system:
  - 30% retrieval score (cosine similarity)
  - 30% reranker score (keyword overlap)
  - 40% LLM self-evaluation (model-based groundedness check)
- **Web API**: FastAPI endpoints for querying and document uploads
- **Thread Safety**: Reentrant locking ensures safe concurrent access
- **Background Indexing**: Automatic document watcher for folder monitoring
- **Image Extraction**: Converts PDF pages to images for visual processing

## Project Structure

```
RAG/
├── src/                           # Main source code
│   ├── main.py                   # FastAPI app with thread-safe AppState
│   ├── config.py                 # Configuration management
│   ├── schemas/
│   │   └── models.py             # Pydantic models (QueryRequest, etc.)
│   ├── ingestion/
│   │   ├── pdf_loader.py         # PDF text/page extraction
│   │   └── image_extractor.py    # PDF to image conversion with OCR
│   ├── chunking/
│   │   └── chunker.py            # Document chunking logic
│   ├── embeddings/
│   │   └── embedder.py           # HuggingFace embeddings
│   ├── retrieval/
│   │   └── retriever.py          # Vector similarity search with scoring
│   ├── generation/
│   │   └── llm.py                # LLM answer generation + confidence
│   ├── automation/
│   │   └── scheduler.py          # Background DocumentWatcher
│   └── templates/
│       └── index.html            # Web UI template
├── tests/
│   ├── test_chunking.py          # Unit tests for chunking
│   ├── test_retrieval.py         # Unit tests for retrieval
│   └── test_integration.py       # Integration tests
├── data/
│   ├── incoming/                 # Drop PDFs here for processing
│   ├── processed/                # Processed documents
│   └── chroma_db/                # ChromaDB vector store
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── bottest.py                    # Example usage script
├── completePipeline_kodecl.py    # Full pipeline example
├── implementation_plan.md        # Architecture documentation
└── walkthrough.md                # Implementation changes summary
```

## Installation

### Prerequisites
- Python 3.9+
- Ollama (for local LLM inference)
- Tesseract (for OCR - optional, for image text extraction)

### Setup

1. **Clone/Extract the project**
   ```powershell
   cd RAG
   ```

2. **Create virtual environment**
   ```powershell
   python -m venv venv
   ```

3. **Activate virtual environment**
   
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   .\venv\Scripts\activate.bat
   ```
   
   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Environment Configuration** (Optional)
   
   Create a `.env` file in the project root:
   ```env
   LOG_FILE=logs/pipeline.log
   OLLAMA_BASE_URL=http://localhost:11434
   EMBEDDING_MODEL=nomic-embed-text
   OLLAMA_MODEL=qwen2.5:3b
   CHUNK_SIZE=512
   CHUNK_OVERLAP=50
   ```

## Usage

### Starting the Web Server

```powershell
python -m src.main
```

The API will be available at `http://localhost:8000`

With the webserver running, you can interact with the RAG system directly in your browser using the built-in web interface and API docs.

**Interactive API docs**: `http://localhost:8000/docs`

### API Endpoints

#### Query Endpoint
```bash
POST /query
Content-Type: application/json

{
  "query": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "sources": [
    {
      "document": "file.pdf",
      "page": 1,
      "text": "..."
    }
  ],
  "confidence": 0.87,
  "retrieval_score": 0.85,
  "reranker_score": 0.90,
  "llm_eval_score": 0.85
}
```

#### Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: <PDF file>
```

#### Rebuild Index
```bash
POST /rebuild-index
```

Rebuilds the ChromaDB index from all processed documents.

### Python Usage

```python
from src.config import Config
from src.ingestion.pdf_loader import PDFLoader
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.llm import LLMGenerator

# Initialize components
config = Config()
pdf_loader = PDFLoader(config)
chunker = Chunker(config)
retriever = Retriever(config)
llm_generator = LLMGenerator(config)

# Load and process a PDF
documents = pdf_loader.load_pdf("document.pdf")
chunks = chunker.chunk(documents)

# Build index and query
retriever.build_index(chunks)
result = llm_generator.generate(
    query="Your question here",
    context_nodes=retriever.retrieve("Your question here", top_k=5)
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Testing

Run all tests:
```powershell
python -m unittest discover -s tests
```

Run specific test module:
```powershell
python -m unittest tests.test_retrieval
```

All tests should pass:
```
Ran 10 tests in 0.041s
OK
```

## Architecture Highlights

### Thread Safety
- **AppState Class**: Encapsulates the index and a reentrant lock (`threading.RLock`)
- All index read/write operations are wrapped with `app_state.lock`
- LLM generation runs outside the lock to prevent blocking concurrent requests

### Confidence Scoring System
```
Confidence = 0.3 × Retrieval_Score + 0.3 × Reranker_Score + 0.4 × LLM_Eval_Score
```

- **Retrieval Score**: Average cosine similarity of retrieved chunks
- **Reranker Score**: Keyword overlap between query and context
- **LLM Self-Evaluation**: Model-based assessment of answer groundedness

### Background Processing
- **DocumentWatcher**: Monitors the `data/incoming/` folder for new PDFs
- Automatically processes and indexes new documents
- Thread-safe updates via `AppState.lock`

## Configuration

Edit [src/config.py](src/config.py) to customize:

- `DATA_DIR`: Directory for data storage
- `CHUNK_SIZE`: Document chunk size (default: 512)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `EMBEDDING_MODEL`: HuggingFace model identifier
- `OLLAMA_MODEL`: Model name in Ollama (default: "llama2")
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

## Dependencies

### Core RAG & Vector Search
- `llama-index` - RAG framework
- `chromadb` - Vector database
- `llama-index-embeddings-huggingface` - Embedding model

### Document Processing
- `pypdf` - PDF text extraction
- `unstructured` - Advanced document parsing
- `pdf2image` - Convert PDFs to images
- `Pillow` - Image processing
- `pytesseract` - OCR for images

### LLM & Web
- `ollama` - Local LLM inference
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

## Troubleshooting

### Ollama not connecting
Ensure Ollama is running:
```bash
ollama serve
```

Then install the models used by this project:
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

If you want to use a smaller model for faster local inference, choose a smaller Ollama model such as `llama2-mini`.

### PDF processing fails
Install Tesseract for OCR:
- **Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

### Vector store issues
Delete the corrupted database and rebuild:
```bash
rm -r data/chroma_db/
# Then POST /rebuild-index endpoint or restart the server
```

### Thread-safety issues
Check logs in `logs/pipeline.log` for locking errors. Ensure all index modifications use `app_state.lock`.

## Performance Optimization

- **Batch Uploads**: Upload multiple PDFs for faster processing
- **Chunking Parameters**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for your document type
- **Embedding Model**: Use smaller models for faster embeddings; larger models for better quality
- **LLM Model**: Balance speed vs. quality with different Ollama models

## Known Limitations

- Requires local Ollama setup for LLM inference
- OCR quality depends on PDF image quality
- Large PDF files may require extended processing time
- Vector database grows with document volume

## Future Enhancements

- [ ] Web UI improvements
- [ ] Multi-language support
- [ ] Advanced chunking strategies (semantic, hierarchical)
- [ ] Query expansion techniques
- [ ] Citation generation
- [ ] Batch query processing
- [ ] Metrics dashboard

## License

[Add your license here]

## Support

For issues, questions, or suggestions, please refer to the documentation files:
- [Implementation Plan](implementation_plan.md) - Architecture details
- [Walkthrough](walkthrough.md) - Implementation changes
