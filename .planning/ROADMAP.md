# Roadmap — ML Paper Reproducibility Predictor

## Milestone 1 (Current)

**Goal:** Working MVP with all core ML components

**Status:** 50% Complete (6/12 stories done)

---

## Phase 1: Complete ML Pipeline

**Goal:** SciBERT classifier + SHAP explainer working end-to-end

**Scope:**
| Story | Description | Status | Effort |
|-------|-------------|--------|--------|
| US-004 | SciBERT Classifier Training | 🔲 Pending | ~2h (needs GPU) |
| US-007 | SHAP Explainability Module | 🔲 Pending | ~1.5h |

**Plans:** 3 plans in 3 waves

Plans:
- [ ] 01-01-PLAN.md — SciBERT classifier training with sliding window
- [ ] 01-02-PLAN.md — SHAP explainability module with caching
- [ ] 01-03-PLAN.md — Integration tests and full pipeline verification

**Requirements:** FR-001, FR-003, US-004, US-007

**Dependencies:**
- US-004 requires trained model for US-007

**UAT Criteria:**
- SciBERT AUROC > 0.78 on test set
- SHAP returns top-5 sentences with scores
- Full analysis (classify + gaps + hints + explain) works

**Exit Criteria:**
- All 4 core modules tested together
- `tests/test_integration.py` passes with SHAP

---

## Phase 2: API + Frontend

**Goal:** Web interface calling backend API

**Scope:**
| Story | Description | Status | Effort |
|-------|-------------|--------|--------|
| US-009 | FastAPI Backend | 🔲 Pending | ~2h |
| US-010 | Streamlit Frontend | 🔲 Pending | ~3h |

**Dependencies:**
- Phase 1 complete (needs all ML modules)

**UAT Criteria:**
- Upload PDF → see score, gaps, hints, highlights
- API responds < 30s
- Works on mobile viewport

**Exit Criteria:**
- Demo video recorded
- localhost end-to-end works

---

## Phase 3: Deploy + Document

**Goal:** Live public demo with documentation

**Scope:**
| Story | Description | Status | Effort |
|-------|-------------|--------|--------|
| US-011 | HuggingFace Spaces Deployment | 🔲 Pending | ~1h |
| US-012 | Documentation & Paper Draft | 🔲 Pending | ~3h |

**Dependencies:**
- Phase 2 complete (needs working app)

**UAT Criteria:**
- Public URL works without login
- Cold start < 60s
- README has architecture diagram

**Exit Criteria:**
- Live demo URL shared
- Paper draft ready for feedback

---

## Completed Work

### Already Done (Pre-Milestone)
- ✅ US-001: Data Pipeline Setup
- ✅ US-002: PDF Extractor Module  
- ✅ US-003: Baseline Classifier (TF-IDF + LR)
- ✅ US-005: NeurIPS Checklist JSON
- ✅ US-006: Gap Detector Module
- ✅ US-008: Fix Hint Generator

### Key Files Created
- `src/pdf_extractor.py` — PyMuPDF methods extraction
- `src/data_pipeline.py` — arXiv fetch, clean, split
- `src/classifier.py` — BaselineClassifier + ReproducibilityClassifier scaffold
- `src/gap_detector.py` — MiniLM semantic matching
- `src/hint_generator.py` — LLM/template hints
- `data/checklist/neurips_checklist.json` — 43 items
- `data/processed/{train,val,test}.parquet` — 22/4/4 samples

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| SciBERT doesn't beat baseline | High | Try lr=1e-5, check data quality |
| GPU not available for training | Medium | Use Google Colab free tier |
| HF API rate limits | Low | Template fallback already works |
| Cold start too slow | Medium | Optimize model loading, lazy init |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| Prior | Use MiniLM for gaps | Fast, good embeddings |
| Prior | Template fallback for hints | Works without API key |
| Now | GSD + Ralph hybrid | GSD for planning, Ralph for execution |
