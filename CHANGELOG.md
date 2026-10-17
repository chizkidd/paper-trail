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
- Edit app.py
- Edit app.py
- Edit app.py
- Edit app.py and requirements.txt
- Edit README
- Edit README
- Edit README
- Edit README
- Edit README
- Improve retrieval and generation quality
- Fix extractive sentence selection to use semantic similarity
- Edit app.py
- Add app.py
- Edit app.py
- Edit app.py (fixing retrieval stage)
- Edit app.py
- Edit app.py
- Edit app.py
- Streamline & clean up app.py
- Update README with new improvements to app.py
- Update README
- Edit README
- Edit README
- Edit README
- Move experimental files to /dev
- Clean modularization: single app.py file into data ingestion, retrieval. LLM backends, utlilities folders
- Edit files
- Edit files
- Edit files
- Edit files
- Edit app.py
- Use newer streamlit version in requirments.txt
- Fix invisible text in sidebar file uploader button and filename
- Update README
- Update README

### Refactoring
- Q_emb pipeline, embedder property, dedup active_doc, markdown history
- Decouple qa.py from Streamlit
