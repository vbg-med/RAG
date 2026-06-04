Write-Host "🚀 RAG Pipeline Setup (PowerShell)" -ForegroundColor Cyan

# 1. Check Python
$pythonInstalled = $false
try {
    $ver = & python --version 2>&1
    if ($ver -match "Python 3.11") {
        $pythonInstalled = $true
        Write-Host "✅ Found: $ver" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Python version is not 3.11 (Found: $ver). Attempting anyway..." -ForegroundColor Yellow
        $pythonInstalled = $true
    }
} catch {
    Write-Host "❌ Python is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# 2. Create venv
Write-Host "📦 Creating virtual environment..." -ForegroundColor Green
& python -m venv .venv

# 3. Create directories
Write-Host "📁 Creating data directories..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "data/incoming", "data/processed", "data/chroma_db", "logs" | Out-Null

# 4. Create .env
Write-Host "🔐 Creating .env file..." -ForegroundColor Green
$envContent = @"
# Paths
DATA_DIR=./data
CHROMA_DB_PATH=./data/chroma_db
LOG_FILE=./logs/pipeline.log

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
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
ONLY_TEXT_EMBEDDING=true
USE_IMAGE_CAPTIONS=true
IMAGE_MODEL=llava
"@

Set-Content -Path ".env" -Value $envContent

# 5. Install requirements in virtualenv
Write-Host "📦 Upgrading pip & installing dependencies..." -ForegroundColor Green
& .venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Install Ollama: https://ollama.ai"
Write-Host "2. Pull models: ollama pull qwen2.5:3b && ollama pull all-minilm:l6-v2"
Write-Host "3. Start Ollama: ollama serve"
Write-Host "4. In another terminal, run: .venv\Scripts\activate && python -m src.main"
