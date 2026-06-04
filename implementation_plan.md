# Implementation Plan - RAG Pipeline Architecture & Quality Improvements

This plan outlines architectural enhancements and bug fixes to improve the stability, thread-safety, and accuracy of the local RAG pipeline.

## User Review Required

> [!IMPORTANT]
> - **Thread Safety & Performance**: We use a Reentrant Lock (`threading.RLock`) to ensure thread safety across API requests and the background folder watcher thread. To prevent blocking the web server during slow LLM inference, LLM generation and self-evaluation are executed outside the state lock.
> - **Self-Evaluation Call**: The confidence scorer will perform a second, lightweight call to the Ollama model for self-evaluation. This ensures a reasoning-based groundedness check but adds slight response latency, which is expected for production RAG pipelines.

## Proposed Changes

---

### Ingestion & Configuration

#### [MODIFY] [main.py](file:///c:/Users/Vignesh/Desktop/Learning/RAG/src/main.py)

- Fix the config attribute name bug: change `config.data_dir` to `config.DATA_DIR`.
- Remove the unused duplicate instantiation of `Embedder(config)`.
- Introduce `AppState` class which encapsulates the `index` and a reentrant lock `lock` for thread safety.
- Replace global `index` with `app_state` instance of `AppState`.
- Modify endpoints (`/query`, `/upload`, `/rebuild-index`) to retrieve/update the index thread-safely using `app_state.lock`.
- Instantiate the background `DocumentWatcher` with `app_state` instead of get/set callbacks.

#### [MODIFY] [scheduler.py](file:///c:/Users/Vignesh/Desktop/Learning/RAG/src/automation/scheduler.py)

- Update `DocumentWatcher` constructor to accept `state` instead of callbacks.
- Update `_process_and_index` to fetch/update `state.index` thread-safely using `state.lock`.

---

### Retrieval & Confidence Scoring

#### [MODIFY] [retriever.py](file:///c:/Users/Vignesh/Desktop/Learning/RAG/src/retrieval/retriever.py)

- In `retrieve`, attach the similarity score to each returned node's metadata as `"retrieval_score"` and as a `score` attribute on the node, keeping the returned signature backward-compatible.

#### [MODIFY] [llm.py](file:///c:/Users/Vignesh/Desktop/Learning/RAG/src/generation/llm.py)

- Replace the naive stop-word overlap confidence score.
- Implement `_calculate_reranker_score` to compute keyword overlap between the query and retrieved context chunks (lexical relevance).
- Implement `_llm_self_evaluation` to query Ollama for a deterministic confidence score (0.0 to 1.0) assessing how well the answer is supported by the contexts.
- Combine these components into a weighted sum: `0.3 * retrieval_score + 0.3 * reranker_score + 0.4 * llm_eval_score`.

---

### Tests

#### [MODIFY] [test_integration.py](file:///c:/Users/Vignesh/Desktop/Learning/RAG/tests/test_integration.py)

- Update references to `src.main.index` to `src.main.app_state.index`.

---

## Verification Plan

### Automated Tests
- Run the test suite:
  ```powershell
  & ".\venv\Scripts\python.exe" -m unittest discover -s tests
  ```

### Manual Verification
- Start the server using Uvicorn:
  ```powershell
  & ".\venv\Scripts\python.exe" -m src.main
  ```
- Trigger a file upload to verify watcher thread safety and index update.
- Query the `/query` endpoint and check the returned answer, sources, and detailed confidence score logs.
