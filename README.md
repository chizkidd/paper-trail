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
- **Pickle-free disk cache**: embeddings and index persisted as JSON + `.npy` + `.npz` — no arbitrary code execution risk
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
| **Structured (no LLM)** | Grounded extractive answer — no hallucination, fully deterministic |
| **Local (Ollama)** | Local LLM synthesis from retrieved evidence (`llama3.1:8b` by default) |
| **Hugging Face** | Serverless Mistral-7B inference with extractive fallback |

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
TFIDF_MAX_FEATURES = 30_000
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 30
HF_MODEL           = "mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_MODEL       = "llama3.1:8b"
OLLAMA_URL         = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
PDF_MAX_BYTES      = 50 * 1024 * 1024   # 50 MB
CACHE_DIR          = ~/.papertrail_cache
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use a **Hugging Face token** (avoids public rate limits), create `.streamlit/secrets.toml`:

```toml
HF_TOKEN = "hf_..."
```

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
| CI               | GitHub Actions — pytest, ruff, pip-audit |

---

## CI / Quality

GitHub Actions runs three jobs on every push and pull request:

| Job | What it checks |
|-----|----------------|
| **Test** | pytest on Python 3.11 and 3.12 |
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
