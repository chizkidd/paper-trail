---
title: Papertrail
emoji: 📄
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Papertrail

![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=chizkidd.paper-trail)

_Ask questions. Get grounded answers from any document you bring._

Papertrail is a lightweight **document question-answering agent** built with Streamlit. It indexes a document, retrieves relevant evidence, and produces grounded answers with clear attribution. No hidden training data. Every answer is derived from the document you load.

---

## Features

- **PDF Upload**: robust text extraction using PyMuPDF (with OCR fallback for scanned PDFs)
- **URL Input**: scrape and index webpage content
- **Paste Text**: index arbitrary text instantly
- **Multi-document support**: load multiple documents and switch between them
- Hybrid retrieval (semantic embeddings + TF-IDF)
- Cross-encoder reranking for precision
- Evidence-based answer generation
- Optional local LLM responses via **Ollama**
- Extractive fallback when generation is unavailable
- Section and page-level attribution
- Supporting passages viewer
- **Conversation export**: download the full Q&A session as a Markdown file
- **Clear chat**: reset the conversation without removing the indexed document
- **Error boundaries**: friendly in-app error messages instead of raw Python tracebacks
- **Thread-safe streaming**: batched word rendering with no `sleep()` calls, server thread never blocked
- **Pickle-free disk cache**: embeddings and index persisted as JSON + `.npy` + `.npz`, no arbitrary code execution risk
- **Crash-safe atomic writes**: cache files written via `.tmp` → `os.replace()` so a crash mid-write never corrupts the index

---

## Architecture

```
Document
↓
Cache lookup ──── hit ──→ load from disk
     ↓ miss
Text extraction (PyMuPDF / requests+BS4)
↓
Chunking (overlapping windows)
↓
Hybrid retrieval (dense embeddings + TF-IDF)
↓
Cross-encoder reranking (MS MARCO MiniLM)
↓
Evidence extraction (MMR sentence selection)
↓
Answer generation (optional LLM)
↓
Cache save
```

---

## Core Components

### 1. Document Parsing

Papertrail extracts document text using **PyMuPDF**, which preserves reading order and spacing reliably. Each paragraph is mapped to its **section heading** (if detected) and **page number** for precise attribution. Scanned PDFs fall back to OCR via pytesseract when available.

### 2. Chunking

The document is split into overlapping text chunks so contextual relationships are preserved across chunk boundaries.

### 3. Hybrid Retrieval

Each chunk is indexed two ways:

**Dense semantic embeddings** (`sentence-transformers/all-MiniLM-L6-v2`): captures semantic similarity.

**Sparse keyword matching** (TF-IDF with bigrams): captures exact phrases and technical terminology.

Combined score:

```python
score = α * dense_similarity + (1 - α) * tfidf_similarity   # α = 0.45 by default
```

### 4. Cross-Encoder Reranking

Top candidates are reranked using **cross-encoder/ms-marco-MiniLM-L-6-v2**. Unlike embedding similarity, cross-encoders evaluate the question and chunk jointly, significantly improving ranking precision.

### 5. Evidence Extraction

Maximal Marginal Relevance (MMR) selects a diverse set of high-relevance sentences from the top chunks, reducing noise and preventing redundancy.

### 6. Answer Modes

| Mode | Description |
|------|-------------|
| **Document quotes** (default) | 1–2 sentences copied verbatim from the document; no LLM, no hallucination, deterministic |
| **AI-written · Groq** | Groq's free tier (`openai/gpt-oss-20b` by default; auto-switches if Groq rejects the model) writes a short answer; needs `GROQ_API_KEY` |
| **AI-written · Ollama** | Local LLM (`llama3.1:8b` by default) writes a short answer from the retrieved evidence |
| **AI-written · Hugging Face** | Serverless Mistral-7B writes a short answer from the retrieved evidence |

AI-written answers pass a faithfulness guard (`papertrail/llm/guard.py`): every sentence must match an evidence
sentence by meaning (cosine ≥ 0.6) and by vocabulary (≥ 60% shared words). If any sentence fails, or the LLM is
unreachable, the answer is replaced by the verbatim document quote.

### 7. Pickle-free Cache (`papertrail/cache.py`)

Each indexed document is persisted to `~/.papertrail_cache/` as four safe files keyed by content hash:

| File | Format | Contents |
|------|--------|----------|
| `<hash>_meta.json` | JSON | chunks, section map, page map |
| `<hash>_embeddings.npy` | numpy binary | dense embedding matrix |
| `<hash>_matrix.npz` | scipy sparse | TF-IDF matrix |
| `<hash>_vocab.json` | JSON | TF-IDF vocabulary + IDF weights |

None of these formats can execute code (unlike pickle). Writes are atomic: data lands in `*.tmp` files first, then `os.replace()` moves them into place.

### 8. Central Config (`papertrail/config.py`)

All tunable hyperparameters live in one file:

```python
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_ALPHA       = 0.45
TFIDF_MAX_FEATURES = 30000
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 30
HF_MODEL           = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
OLLAMA_MODEL       = "llama3.1:8b"
OLLAMA_URL         = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
PDF_MAX_BYTES      = 50 * 1024 * 1024   # 50 MB
CACHE_DIR          = ~/.papertrail_cache
```

### 9. Session Reliability

**Model warm-up**: embedder and cross-encoder are loaded at app startup via `@st.cache_resource`. The first question sees no cold-start delay.

**Session state validation**: `_validate_session_state()` runs on every render and repairs corrupt or missing document entries before any rendering occurs.

**Error boundaries**: the answer pipeline is wrapped in a `try/except` block that catches `MemoryError` and general exceptions, showing a user-friendly message with an error detail caption instead of a raw traceback.

**Pure-Python core**: `papertrail/qa.py` has no Streamlit dependency. All public functions (`get_documents`, `get_active_doc`, `get_active_retriever`, `build_knowledge_base`, `remove_document`, `answer_question`) accept plain Python arguments and return plain Python values. Session state is owned entirely by `app.py`, making the module importable and testable without a running Streamlit server.

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The **AI-written · Groq** mode is the easiest free option: create a key at
[console.groq.com](https://console.groq.com) and add it to `.streamlit/secrets.toml` (or the app's Secrets):

```toml
GROQ_API_KEY = "gsk_..."
```

Override the model with the `GROQ_MODEL` env var. Free tiers are rate limited; when the quota is hit the answer tag
says so and the answer falls back to a document quote.

The **AI-written · Hugging Face** mode calls the Hugging Face Inference Providers router
(`router.huggingface.co`) and **requires** a token with the *Make calls to Inference Providers*
permission. Create `.streamlit/secrets.toml` (or the app's Secrets on Streamlit Cloud):

```toml
HF_TOKEN = "hf_..."
```

Free accounts get a small monthly credit allowance; when it runs out the answer tag reads
`quote · AI unavailable (HF free credits used up or rate limited)`. To use a different model, set the
`HF_MODEL` env var to any model served by an inference provider. **Ollama** mode only works where an
Ollama server is reachable (see below), not on Streamlit Cloud.

To use a **remote Ollama instance**, set the env var:

```bash
export OLLAMA_URL="http://your-server:11434/api/generate"
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Add `HF_TOKEN` under **Secrets** in the app settings (optional)
5. Deploy

---

## Tech Stack

| Component        | Tool                           |
|------------------|--------------------------------|
| UI               | Streamlit                      |
| Embeddings       | sentence-transformers          |
| Sparse retrieval | scikit-learn TF-IDF            |
| Neural reranking | CrossEncoder (MS MARCO MiniLM) |
| PDF parsing      | PyMuPDF                        |
| Web scraping     | requests + BeautifulSoup       |
| Local LLM        | Ollama                         |
| Cache            | numpy `.npy`, scipy `.npz`, JSON (pickle-free) |
| CI               | GitHub Actions: pytest, ruff, pip-audit |

---

## CI / Quality

GitHub Actions runs three jobs on every push and pull request:

| Job | What it checks |
|-----|----------------|
| **Test** | pytest: 86 tests across 6 files, Python 3.11 and 3.12 |
| **Lint** | ruff (`E`, `F`, `W` rules) |
| **Security** | pip-audit dependency vulnerability scan |

---

## Extending Papertrail

Possible future improvements:

- [ ] Multi-document search across all loaded documents
- [ ] Vector database backend (FAISS, Qdrant)
- [ ] Structured citations
- [ ] Conversation memory
- [ ] Additional document formats (`.docx`, `.csv`)
- [ ] Improved OCR for scanned PDFs

---

## Philosophy

> The model should answer from your document, not from its training data. Every answer is grounded in retrieved evidence.
