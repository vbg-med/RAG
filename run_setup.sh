#!/bin/bash

echo "🚀 RAG Pipeline Setup"

# 1. Check Python
python3.11 --version || { echo "❌ Python 3.11 not found. Install it first."; exit 1; }

# 2. Create venv
echo "📦 Creating virtual environment..."
python3.11 -m venv .venv

# 3. Activate venv
source .venv/bin/activate

# 4. Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 5. Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# 6. Create directories
echo "📁 Creating data directories..."
mkdir -p data/incoming data/processed data/chroma_db logs

# 7. Create .env
echo "🔐 Creating .env file..."
cat > .env << 'EOF'
# Paths
DATA_DIR=./data
CHROMA_DB_PATH=./data/chroma_db
LOG_FILE=./logs/pipeline.log

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
OLLAMA_EMBED_MODEL=all-minilm:l6-v2

# LLM settings
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1024

# RAG settings
CHUNK_SIZE=600
CHUNK_OVERLAP=100
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.5

# Image processing
USE_IMAGE_CAPTIONS=true
IMAGE_MODEL=llava
EOF

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Install Ollama: https://ollama.ai"
echo "2. Pull models: ollama pull mistral && ollama pull nomic-embed-text"
echo "3. Start Ollama: ollama serve"
echo "4. In another terminal, run: python -m src.main"
