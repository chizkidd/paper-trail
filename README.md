# Papertrail

Ask questions. Get grounded answers from any document you bring.

Papertrail is a lightweight **document question-answering agent** built with Streamlit.  
It indexes a document, retrieves relevant evidence, and produces grounded answers with clear attribution.

No hidden training data.  
Every answer is derived from the document you load.

---

## Features

- **PDF Upload**: robust text extraction using PyMuPDF
- **URL Input**: scrape and index webpage content
- **Paste Text**: index arbitrary text instantly
- Hybrid retrieval (semantic embeddings + TF-IDF)
- Cross-encoder reranking for precision
- Evidence-based answer generation
- Optional local LLM responses via **Ollama**
- Extractive fallback when generation is unavailable
- Section and page-level attribution
- Supporting passages viewer

---

## Architecture

```

Document
↓
Text extraction
↓
Chunking
↓
Hybrid retrieval (embeddings + TF-IDF)
↓
Cross-encoder reranking
↓
Evidence extraction (MMR)
↓
Answer generation (optional)

```

Pipeline summary:

```

Chunking → Hybrid retrieval → Neural rerank → Evidence grounding → Optional generation

```

---

## 1. Document Parsing

Papertrail extracts document text using **PyMuPDF**, which preserves reading order and spacing more reliably than many PDF parsers.

Each paragraph is mapped to:

- its **section heading** (if detected)
- its **page number**

These mappings enable precise attribution in answers.

---

## 2. Chunking

The document is split into overlapping text chunks so that contextual relationships are preserved across chunk boundaries.

Chunking enables efficient indexing and retrieval across large documents.

---

## 3. Hybrid Retrieval

Each chunk is indexed in two ways:

### Dense semantic embeddings
Using `sentence-transformers`.

This captures semantic similarity.

Example:

```

advantages ≈ pros
drawbacks ≈ disadvantages

````

### Sparse keyword matching
Using TF-IDF with bigrams.

This captures exact phrases and technical terminology.

The combined score:

```python
score = α * dense_similarity + (1 - α) * tfidf_similarity
````

Hybrid retrieval improves recall across both semantic and lexical queries.

---

## 4. Cross-Encoder Reranking

The top candidate chunks are reranked using a **cross-encoder (MS MARCO MiniLM)**.

Unlike embedding similarity, cross-encoders evaluate the **question and chunk jointly**, significantly improving ranking precision.

---

## 5. Evidence Extraction

Maximal Marginal Relevance (MMR) selects a diverse set of high-relevance sentences from the top chunks.

This produces an **evidence pack** that:

* reduces noise
* prevents redundancy
* ensures answers remain grounded

---

## 6. Answer Modes

Papertrail supports three answer modes.

### Structured (no LLM)

Produces a grounded extractive answer directly from the document.

This mode guarantees:

* no hallucination
* deterministic behavior
* complete grounding in the source text

---

### Local (Ollama)

Uses a local LLM to synthesize an answer from retrieved evidence.

Advantages:

* higher fluency
* better explanations
* no external API required

Install Ollama:

```
https://ollama.ai
```

Then pull a model:

```bash
ollama pull llama3
```

---

### Hugging Face (best effort)

Uses serverless inference for generation when available.

If generation fails or times out, Papertrail automatically falls back to grounded extractive answers.

---

## Attribution

Each answer can include:

* detected **section headings**
* **PDF page numbers**
* **supporting passages**

This makes it easy to verify exactly where an answer came from.

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set `app.py` as the entry point
5. Deploy

---

## Tech Stack

| Component        | Tool                           |
| ---------------- | ------------------------------ |
| UI               | Streamlit                      |
| Embeddings       | sentence-transformers          |
| Sparse retrieval | scikit-learn TF-IDF            |
| Neural reranking | CrossEncoder (MS MARCO MiniLM) |
| PDF parsing      | PyMuPDF                        |
| Web scraping     | requests + BeautifulSoup       |
| Local LLM        | Ollama                         |

---

## Why Hybrid Retrieval?

Pure TF-IDF fails on synonyms.

Example:

```
pros and cons
advantages and disadvantages
```

Semantic retrieval fixes this, but pure embeddings miss exact keywords.

Hybrid retrieval combines both.

---

## Extending Papertrail

Possible future improvements:

- [ ] Multi-document search
- [ ] Vector database backend (FAISS, Qdrant)
- [ ] Persistent embedding cache
- [ ] Structured citations
- [ ] Conversation memory
- [ ] Additional document formats (`.docx`, `.csv`)
- [ ] Improved OCR for scanned PDFs

---

## Philosophy

Papertrail follows one principle:

> The model should answer from your document, not from its training data. Every answer is grounded in retrieved evidence.

