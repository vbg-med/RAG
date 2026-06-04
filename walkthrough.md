# Walkthrough - RAG Pipeline Architectural Improvements

This walkthrough summarizes the structural, safety, and algorithm enhancements implemented in the RAG pipeline.

## Changes Made

### 1. Bug Fixes & Refactoring
- **Configuration Attribute Bug**: Fixed `config.data_dir` to `config.DATA_DIR` in `src/main.py`.
- **Double Embedder Creation**: Removed the redundant unused `embedder = Embedder(config)` instantiation in `src/main.py`.

### 2. Thread Safety & App State
- **AppState Object**: Created a thread-safe `AppState` class in `src/main.py` containing `index` and a reentrant lock (`threading.RLock`).
- **Dependency Injection**: Refactored background `DocumentWatcher` in `src/automation/scheduler.py` to receive `AppState` via dependency injection.
- **Locking Scheme**: Wrapped index read/write operations (e.g. index loading, index updates/building, node retrieval, background indexing) inside `AppState.lock`. Slow LLM answer generation and self-eval runs outside the lock to prevent blocking concurrent API requests.

### 3. High-Quality Confidence Scoring
- **Retriever Updates**: Modified `Retriever.retrieve` in `src/retrieval/retriever.py` to attach the similarity score of retrieved nodes directly to the returned node objects (as properties and in metadata `"retrieval_score"`), keeping the API signature fully backward-compatible.
- **Upgraded Scorer**: Upgraded `llm_generator.generate` in `src/generation/llm.py` to compute confidence as a weighted combination:
  $$\text{Confidence} = 0.3 \times S_{\text{retrieval}} + 0.3 \times S_{\text{reranker}} + 0.4 \times S_{\text{llm\_eval}}$$
  - **Retrieval Score**: Average cosine similarity score of context chunks.
  - **Reranker Score**: Custom lexical overlap score calculating question keyword matching ratios.
  - **LLM Self-Evaluation**: Asks Ollama model to evaluate answer truthfulness/groundedness against context with a fallback word-level overlap check in case of LLM query errors.

### 4. Tests Update
- Updated `tests/test_integration.py` to set/assert index status on `src.main.app_state.index` instead of the legacy `src.main.index`.

---

## Verification Results

We verified all changes using the Python unit tests within the project:

```powershell
& ".\venv\Scripts\python.exe" -m unittest discover -s tests
```

Output:
```text
C:\Users\Vignesh\Desktop\Learning\RAG\venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
..........
----------------------------------------------------------------------
Ran 10 tests in 0.041s

OK
```

All 10 tests executed successfully and passed, verifying that both unit and integration tests are clean and correct.
