"""FastAPI app with Uvicorn + RAG pipeline"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import pipeline modules
from src.config import Config
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.image_extractor import ImageExtractor
from src.chunking.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.llm import LLMGenerator
from src.automation.scheduler import DocumentWatcher
from src.schemas.models import QueryRequest, QueryResponse, UploadResponse

# Setup logging
logging.basicConfig(
    filename=os.getenv("LOG_FILE", "logs/pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG Pipeline API",
    description="Local RAG system for PDF document QA",
    version="1.0.0",
)

# Initialize pipeline components
config = Config()
pdf_loader = PDFLoader(config)
image_extractor = ImageExtractor(config)
chunker = Chunker(config)
embedder = Embedder(config)
retriever = Retriever(config)
llm_generator = LLMGenerator(config)

# Global state
index = None  # Will be loaded/rebuilt


def get_index():
    global index
    return index


def set_index(new_index):
    global index
    index = new_index


# Initialize background watcher
watcher = DocumentWatcher(
    config=config,
    pdf_loader=pdf_loader,
    image_extractor=image_extractor,
    chunker=chunker,
    retriever=retriever,
    get_index_func=get_index,
    set_index_func=set_index,
)


def initialize_index():
    """Load existing index or build from PDFs"""
    global index

    try:
        logger.info("Initializing RAG index...")

        # Check if index can be loaded from Chroma
        loaded_index = retriever.load_index()
        if loaded_index:
            index = loaded_index
            logger.info("Successfully loaded existing index from Chroma DB")
        else:
            logger.info("Building index from PDFs in incoming folder...")
            # Load all PDFs from data/incoming
            incoming_dir = Path(config.data_dir) / "incoming"
            pdf_files = list(incoming_dir.glob("*.pdf"))

            if not pdf_files:
                logger.warning("No PDFs found in data/incoming")
                index = None
                return

            documents = []
            for pdf_file in pdf_files:
                logger.info(f"Processing {pdf_file.name}...")

                # Load text
                text_docs = pdf_loader.load(str(pdf_file))
                documents.extend(text_docs)

                # Extract and caption images
                if config.USE_IMAGE_CAPTIONS:
                    image_docs = image_extractor.process_pdf(str(pdf_file))
                    documents.extend(image_docs)

            if not documents:
                logger.warning("No documents loaded from files.")
                index = None
                return

            # Chunk
            chunks = chunker.chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")

            # Build index
            index = retriever.build_index(chunks)
            logger.info("Index built successfully")

    except Exception as e:
        logger.error(f"Error initializing index: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    try:
        initialize_index()
    except Exception as e:
        logger.error(f"Error in startup indexing: {e}")

    # Start the background folder watcher
    watcher.start()
    logger.info("✅ RAG Pipeline ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background processes on shutdown"""
    watcher.stop()
    logger.info("✅ RAG Pipeline shutdown completed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "index_loaded": index is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline

    Args:
        request: QueryRequest with 'question' field

    Returns:
        Answer with sources and confidence
    """
    if not index:
        raise HTTPException(
            status_code=503, detail="Index not initialized. Upload PDFs first."
        )

    try:
        logger.info(f"Query: {request.question}")

        # Retrieve relevant chunks
        retrieved = retriever.retrieve(
            index=index,
            query=request.question,
            top_k=config.TOP_K_RETRIEVAL,
            threshold=config.SIMILARITY_THRESHOLD,
        )

        if not retrieved:
            return QueryResponse(
                answer="No relevant information found in documents.",
                sources=[],
                confidence=0.0,
            )

        # Generate answer
        answer, confidence = llm_generator.generate(
            question=request.question, context=retrieved
        )

        # Extract sources
        sources = [
            {
                "text": chunk.text[:100] + "...",
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "N/A"),
                "type": chunk.metadata.get("type", "text"),
            }
            for chunk in retrieved
        ]

        logger.info(f"Answer generated. Confidence: {confidence:.2f}")

        return QueryResponse(answer=answer, sources=sources, confidence=confidence)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a new PDF and add to index

    Args:
        file: PDF file to upload

    Returns:
        Upload confirmation with chunk count
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    try:
        logger.info(f"Uploading {file.filename}...")

        # Save to incoming folder
        save_path = Path(config.DATA_DIR) / "incoming" / file.filename
        content = await file.read()
        save_path.write_bytes(content)

        # Process immediately
        text_docs = pdf_loader.load(str(save_path))
        documents = list(text_docs)
        if config.USE_IMAGE_CAPTIONS:
            image_docs = image_extractor.process_pdf(str(save_path))
            documents.extend(image_docs)

        chunks = chunker.chunk_documents(documents)

        # Add to index
        global index
        if index:
            retriever.add_to_index(index, chunks)
        else:
            # Build new index if doesn't exist
            index = retriever.build_index(chunks)

        logger.info(f"✅ Added {len(chunks)} chunks from {file.filename}")

        return UploadResponse(
            filename=file.filename,
            chunks_added=len(chunks),
            message="PDF processed and added to index",
        )

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    try:
        chroma_path = Path(config.CHROMA_DB_PATH)
        pdf_count = len(list(Path(config.DATA_DIR / "incoming").glob("*.pdf")))

        return {
            "pdfs_in_queue": pdf_count,
            "index_exists": index is not None,
            "chroma_db_size_mb": sum(
                f.stat().st_size for f in chroma_path.rglob("*") if f.is_file()
            )
            / 1024
            / 1024
            if chroma_path.exists()
            else 0,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-index")
async def rebuild_index():
    """
    Force rebuild index from all PDFs in data/incoming
    """
    try:
        logger.info("Forcing index rebuild...")
        global index
        index = None
        initialize_index()
        return {"message": "Index rebuilt successfully"}
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run with Uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Hot reload in development
        log_level="info",
    )
