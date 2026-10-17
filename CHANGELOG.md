# Changelog

All notable changes to Papertrail are documented here.

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
