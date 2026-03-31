---
milestone: v1.0
audited: 2026-03-31T05:55:00Z
status: gaps_found
scores:
  user_stories: 6/12
  integration: 6/6
  acceptance_criteria:
    us-001: 7/7
    us-002: 6/6
    us-003: 5/5
    us-005: 4/4
    us-006: 7/7
    us-008: 5/5
gaps:
  user_stories:
    - id: US-004
      title: "SciBERT Classifier Module"
      status: not_started
      priority: 4
      blocker: false
      notes: "Scaffold exists in classifier.py, needs training"
    - id: US-007
      title: "SHAP Explainability Module"
      status: not_started
      priority: 7
      blocker: false
      notes: "Requires trained classifier"
    - id: US-009
      title: "FastAPI Backend"
      status: not_started
      priority: 9
      blocker: false
      notes: "All ML modules ready"
    - id: US-010
      title: "Streamlit Frontend"
      status: not_started
      priority: 10
      blocker: false
      notes: "Requires API"
    - id: US-011
      title: "HuggingFace Deployment"
      status: not_started
      priority: 11
      blocker: false
      notes: "Requires frontend"
    - id: US-012
      title: "Documentation & Paper"
      status: not_started
      priority: 12
      blocker: false
      notes: "Final deliverable"
tech_debt:
  - module: data_pipeline
    items:
      - "Sample dataset only (30 papers), full PwC dataset not downloaded"
      - "arXiv rate limiting may need adjustment for full dataset"
  - module: classifier
    items:
      - "Baseline overfits on small dataset - expected, not a bug"
      - "SciBERT class scaffolded but not trained"
  - module: pdf_extractor
    items:
      - "Methods section detection picks up tables in some PDFs"
      - "May need manual tuning of section header patterns"
---

# Milestone v1.0 Audit Report

## Executive Summary

| Metric | Status |
|--------|--------|
| User Stories Complete | 6/12 (50%) |
| Integration Tests | 6/6 ✓ |
| Critical Blockers | 0 |
| Tech Debt Items | 6 |

## Completed User Stories

### ✓ US-001: Data Pipeline Setup
- **Files:** `src/data_pipeline.py`, `scripts/download_data.py`
- **Output:** `data/processed/{train,val,test}.parquet`
- **Status:** All 7 acceptance criteria met
- **Evidence:** 30 papers processed, 22/4/4 train/val/test split

### ✓ US-002: PDF Extractor Module
- **Files:** `src/pdf_extractor.py`
- **Status:** All 6 acceptance criteria met
- **Evidence:** PyMuPDF extraction, regex section detection, URL support

### ✓ US-003: Baseline Classifier (TF-IDF + LR)
- **Files:** `src/classifier.py`, `models/baseline/`
- **Status:** All 5 acceptance criteria met
- **Evidence:** Model saved, predicts reproducibility scores

### ✓ US-005: NeurIPS Checklist JSON
- **Files:** `data/checklist/neurips_checklist.json`
- **Status:** All 4 acceptance criteria met
- **Evidence:** 43 items with id, category, item, severity, keywords

### ✓ US-006: Gap Detector Module
- **Files:** `src/gap_detector.py`
- **Status:** All 7 acceptance criteria met
- **Evidence:** MiniLM-L6 similarity, threshold 0.35, summary stats

### ✓ US-008: Fix Hint Generator
- **Files:** `src/hint_generator.py`
- **Status:** All 5 acceptance criteria met
- **Evidence:** Template-based hints work, HF API integration ready

## Pending User Stories

| ID | Title | Dependencies | Ready? |
|----|-------|--------------|--------|
| US-004 | SciBERT Classifier | US-003 ✓ | ✓ Ready |
| US-007 | SHAP Explainer | US-004 | Blocked |
| US-009 | FastAPI Backend | US-004, US-006 ✓, US-007, US-008 ✓ | Partial |
| US-010 | Streamlit Frontend | US-009 | Blocked |
| US-011 | HF Deployment | US-010 | Blocked |
| US-012 | Documentation | US-011 | Blocked |

## Cross-Module Integration

| From | To | Status |
|------|----|--------|
| pdf_extractor → data_pipeline | ✓ Working |
| data_pipeline → classifier | ✓ Working |
| classifier → gap_detector | N/A (independent) |
| gap_detector → hint_generator | ✓ Working |
| All modules → API | Not built |
| API → Frontend | Not built |

## Tech Debt

### Data Pipeline
- Sample dataset only (30 papers) - full PwC dataset (~1200) not downloaded
- arXiv rate limiting may need adjustment for production

### Classifier
- Baseline overfits on small dataset (expected behavior)
- SciBERT class scaffolded but requires GPU training

### PDF Extractor
- Methods section detection occasionally picks up tables
- Section header patterns may need tuning per paper format

## Recommendations

1. **Continue with US-004 (SciBERT)** - Core ML blocked items depend on this
2. **Alternatively, build US-009 (API)** - Can integrate baseline + gap detector now
3. **Consider downloading full dataset** before SciBERT training
