# ML Paper Reproducibility Predictor

## Vision

A web tool that helps researchers verify if ML papers are reproducible by analyzing the methods section and providing:
- **Reproducibility Score** (0-100) using fine-tuned SciBERT
- **Gap Report** - missing checklist items via semantic similarity
- **Evidence Highlights** - SHAP-based sentence attribution  
- **Fix Hints** - LLM-generated actionable suggestions

## Problem Statement

ML research suffers from a reproducibility crisis - many papers don't document enough detail to replicate results. Manually checking the NeurIPS reproducibility checklist is tedious. This tool automates that process.

## Target User

- ML researchers wanting to check their own papers before submission
- Reviewers wanting to quickly assess paper reproducibility
- Students learning what makes papers reproducible

## Requirements

### Validated (Already Implemented)

- ✓ PDF extraction with methods section detection — `src/pdf_extractor.py`
- ✓ Data pipeline with arXiv fetching — `src/data_pipeline.py`
- ✓ Baseline TF-IDF classifier — `src/classifier.py`
- ✓ NeurIPS 43-item checklist JSON — `data/checklist/neurips_checklist.json`
- ✓ Gap detector with MiniLM embeddings — `src/gap_detector.py`
- ✓ Fix hint generator with templates — `src/hint_generator.py`

### Active (Remaining Work)

- [ ] US-004: SciBERT classifier (scaffold exists, needs training)
- [ ] US-007: SHAP explainer module
- [ ] US-009: FastAPI backend
- [ ] US-010: Streamlit frontend  
- [ ] US-011: HuggingFace Spaces deployment
- [ ] US-012: Documentation & paper draft

### Out of Scope

- Real-time PDF upload from browser (will use URL/upload)
- LaTeX source analysis (methods section from PDF only)
- Paper writing assistance (detection only, not generation)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SciBERT over BERT | Domain-specific vocabulary | — Pending |
| Ablation SHAP over full SHAP | Faster, sufficient for sentence-level | — Pending |
| MiniLM for gap detection | Good semantic matching, small footprint | ✓ Validated |
| Mistral-7B for hints | Free tier, good instruction following | ✓ Validated |
| Parquet over SQL | Simple, fast at this scale (~1200 rows) | ✓ Validated |
| Streamlit over React | Faster to build, native HF Spaces | — Pending |

## Technical Stack

- **ML**: PyTorch, HuggingFace Transformers, sentence-transformers, SHAP
- **API**: FastAPI, Pydantic
- **Frontend**: Streamlit
- **Data**: Pandas, Parquet, PyMuPDF
- **Deployment**: HuggingFace Spaces, Docker

## Success Metrics

| Metric | Target |
|--------|--------|
| SciBERT AUROC | > 0.78 |
| Gap detection precision | > 0.68 |
| API response time | < 30s |
| Cold start | < 60s |

## Integration Points

- **Ralph (prd.json)**: User story tracking and passes/fails
- **GSD**: Planning, roadmap, phase execution

---
*Last updated: Milestone 1 - 50% complete*
