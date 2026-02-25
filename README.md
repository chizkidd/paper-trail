# Papertrail

Ask questions, get answers from any document you bring.

A document Q&A agent built with Streamlit and scikit-learn. Load any document and ask it questions: no hardcoded knowledge base, no external LLM API required.

## What it does

- **PDF upload**: Drop in any PDF and it extracts text automatically
- **URL**: Paste a webpage URL and it scrapes the content
- **Paste text**: Paste raw text directly
- Chunks the content, indexes it with TF-IDF, and answers questions using cosine similarity
- Shows a confidence score for every answer
- Falls back gracefully when no relevant match is found

## How it works

1. Text is split into overlapping 300-word chunks (50-word overlap to preserve context across boundaries)
2. A TF-IDF vectorizer with bigrams indexes all chunks
3. On each question, cosine similarity ranks the top matching chunks
4. The best chunks are stitched together and returned as the answer
5. Confidence score reflects the cosine similarity of the best match

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `app.py` as the entry point
4. Deploy: you'll get a shareable public URL

## Tech

- **Streamlit** for the UI
- **scikit-learn** TF-IDF + cosine similarity for retrieval
- **pdfplumber** for PDF text extraction
- **beautifulsoup4** for URL scraping
- No external LLM API required

## Extending this

- Swap TF-IDF for `sentence-transformers` embeddings for semantic search
- Add OpenAI/Claude API as a fallback for low-confidence answers
- Support `.docx`, `.csv`, or structured data inputs
- Add multi-document support with source attribution per answer