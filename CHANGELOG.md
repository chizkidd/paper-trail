# Changelog

All notable changes to Papertrail are documented here.

## [0.2.0] - 2026-10-17

### Features
- Answers come from the document by default: new "Document quotes" mode returns 1-2 verbatim sentences, with no LLM and no network needed
- Faithfulness guard for AI-written answers: every sentence must match the document by meaning and vocabulary, otherwise the quote is shown instead
- Per-answer tag showing what actually produced the answer (`quote`, `AI-written`, or the fallback reason), saved with the message and included in the export
- Groq free-tier AI mode (`GROQ_API_KEY`), with automatic model discovery if the configured model is rejected
- Failure reasons for AI mode, e.g. no API key, key rejected, quota used up, model unavailable, timeout

### Bug Fixes
- Removed the misleading "generated/grounded answer" label, which was wrong whenever the fallback ran
- DOCX: headings no longer leak into answers and superscript footnote numbers are dropped
- Answers are capped at 2 sentences, with no inline `Evidence:` suffix and stray footnote digits stripped
- Hugging Face backend moved from the retired `api-inference` endpoint to the Inference Providers router
- Empty or rejected AI streams now replace the bubble in place instead of rendering a second one
- Groq default model changed to `openai/gpt-oss-20b`, with token headroom for reasoning models

### Refactoring
- One shared OpenAI-compatible streaming client for Groq and Hugging Face
- Answer-mode names centralised in `config.py`; non-stream LLM generators removed

## [0.1.0] - 2026-10-14

### Bug Fixes
- Security, reliability, and test coverage improvements
- Resolve ruff lint failures in CI
- Streamlit audit — XSS, atomic cache, secrets, deployment hardening
- Resolve pip-audit CI failures
- White text in assistant bubbles and expanders
- Clean up noisy supporting passages in expander
- Filter Unicode math sentences from supporting passages
- Remove blank space at bottom of page on mobile
- Update remaining get_documents() call in sidebar to pass documents arg
- Bump sentence-transformers and transformers to resolve CVEs

### Documentation
- Update README to reflect all recent improvements
- Update README with post-refactor improvements

### Features
- Multi-doc support, persistence, OCR fallback, Mistral-7B generation
- **`README`:** Add repo visitor count badge
- Production hardening — streaming fix, error boundaries, conversation export
- Close remaining P1/P4 gaps — model warm-up and invariant tests
- Add Railway deployment config
- HF Spaces config, larger chunk overlap and evidence window
- Answer_html cleanup, .docx support, cross-doc search, true LLM streaming

### Other Changes
- Initial app.py — iterative development of retrieval and generation pipeline
- Improve retrieval and generation quality (chunk sizing, dynamic top_k, prompt rewrite, quality gate)
- Fix extractive sentence selection to use semantic similarity over TF-IDF
- Streamline and clean up app.py
- Modularize codebase: split monolithic app.py into ingest, retrieval, LLM backends, and utilities packages
- Move experimental files to /dev
- Fix invisible text in sidebar file uploader button and filename
- Bump Streamlit version in requirements.txt
- Update README across multiple iterations

### Refactoring
- Q_emb pipeline, embedder property, dedup active_doc, markdown history
- Decouple qa.py from Streamlit
