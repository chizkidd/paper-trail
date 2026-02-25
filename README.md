# Papertrail

Ask questions. Get grounded answers from any document you bring.

Papertrail is a document Q&A agent built with Streamlit. It indexes your document, retrieves relevant evidence, and generates grounded answers with clear attribution.

No hardcoded knowledge base. No hidden training data. Every answer is derived from your document.

---

## What it does

- **PDF upload:** Extracts text using `pdfplumber`
- **URL input:** Scrapes webpage content
- **Paste text:** Direct raw text input
- Splits content into overlapping chunks
- Hybrid retrieval (semantic embeddings + TF-IDF)
- Cross-encoder reranking for precision
- MMR-based sentence extraction for grounded answers
- Optional generation via Hugging Face Inference API (FLAN-T5)
- Displays supporting passages and section/page attribution
- Graceful fallback when no relevant match is found

---

## Architecture

### 1. Chunking

The document is split into overlapping chunks to preserve context across boundaries.

### 2. Hybrid Retrieval

Each chunk is indexed two ways:

- **Dense embeddings** using `sentence-transformers`
- **Sparse TF-IDF** using bigrams

A weighted hybrid score combines both:

```python
score = α * semantic_similarity + (1 - α) * keyword_similarity
```

This enables:
- Synonym understanding (pros ≈ advantages)
- Exact keyword precision
- Robust performance across writing styles

### 3. Cross-Encoder Reranking

Top candidates are reranked using a cross-encoder model (MS MARCO MiniLM) for improved semantic accuracy.

### 4. Extractive Grounding (MMR)

Maximal Marginal Relevance selects diverse, high-relevance sentences to form a grounded answer.

### 5. Optional Generation

A lightweight FLAN-T5 model (via Hugging Face Inference API) synthesizes a structured response using only retrieved evidence.

If generation is unavailable or fails, the system falls back to extractive output.

### 6. Attribution

Each answer can include:
- Section headers (when available)
- Page numbers (for PDFs)
- Supporting passages in an expandable panel

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set `app.py` as the entry point
5. Deploy

---

## Tech Stack

* **Streamlit:** UI
* **sentence-transformers:** semantic embeddings
* **scikit-learn:** TF-IDF + cosine similarity
* **CrossEncoder (MS MARCO MiniLM):** reranking
* **pdfplumber:** PDF text extraction
* **BeautifulSoup + requests:** URL scraping
* **FLAN-T5 (Hugging Face Inference API):** optional generation

---

## Why Hybrid Retrieval?

Pure TF-IDF fails on synonyms.

Example:

* "pros and cons"
* "advantages and disadvantages"

Hybrid retrieval solves this while still preserving exact keyword matching.

---

## Extending Papertrail

* Multi-document search with per-source ranking
* Vector store backend (FAISS, Qdrant)
* Persistent embedding cache
* Structured citation formatting
* Conversation memory across turns
* Local LLM deployment instead of remote API
* Support for `.docx`, `.csv`, or structured data inputs

---

## Philosophy

Papertrail is built around one principle:

The model should answer from your document, not from its training data.

Every answer is grounded in retrieved evidence.

