# Requirements — ML Paper Reproducibility Predictor

## Functional Requirements

### FR-001: Classification Pipeline
**Priority:** High | **Status:** Partial

The system must classify papers as reproducible/non-reproducible.

**Acceptance Criteria:**
- SciBERT fine-tuned on Papers With Code dataset
- Sliding window for >512 token texts
- Return score (0-100), label, confidence
- AUROC > 0.78, F1 > 0.74

**Current State:** Baseline (TF-IDF+LR) complete. SciBERT scaffold exists.

---

### FR-002: Gap Detection
**Priority:** High | **Status:** ✓ Complete

The system must identify missing reproducibility checklist items.

**Acceptance Criteria:**
- Semantic similarity matching vs 43 NeurIPS items
- Missing threshold: max_similarity < 0.35
- Report coverage score and severity breakdown
- Precision > 0.68 on manual eval

**Implementation:** `src/gap_detector.py`

---

### FR-003: Evidence Explanation
**Priority:** High | **Status:** Not Started

The system must show which sentences influenced the score.

**Acceptance Criteria:**
- Ablation-based SHAP approximation
- Top 5 most influential sentences highlighted
- Color coding: green (positive), yellow (negative), none (neutral)
- Performance: ~6s for 20 sentences

---

### FR-004: Fix Hints
**Priority:** Medium | **Status:** ✓ Complete

The system must provide actionable suggestions per gap.

**Acceptance Criteria:**
- LLM-generated one-line hints
- Template fallback when no API token
- <2s per hint

**Implementation:** `src/hint_generator.py`

---

### FR-005: Web Interface
**Priority:** High | **Status:** Not Started

Users must interact via web browser.

**Acceptance Criteria:**
- PDF upload or arXiv URL input
- Score gauge visualization
- Gap report table sorted by severity
- SHAP-highlighted text display
- Mobile responsive

---

### FR-006: REST API
**Priority:** High | **Status:** Not Started

Backend must expose analysis via REST.

**Acceptance Criteria:**
- POST /analyze (PDF upload or URL)
- GET /health
- Pydantic validation
- Response < 30s, CORS enabled

---

### FR-007: Deployment
**Priority:** Medium | **Status:** Not Started

Tool must be accessible online.

**Acceptance Criteria:**
- HuggingFace Spaces deployment
- Cold start < 60s
- Public URL, no login required

---

## Non-Functional Requirements

### NFR-001: Performance
- Full pipeline < 30s
- Gap detection < 5s
- SHAP explanation < 10s

### NFR-002: Cost
- $0 operational cost (free tiers only)
- HF Inference API free tier for hints
- HF Spaces free tier for hosting

### NFR-003: Maintainability
- Modular Python structure (src/)
- Type hints throughout
- Unit tests for core modules

---

## Traceability Matrix

| Req | User Story | Phase |
|-----|------------|-------|
| FR-001 | US-003, US-004 | Phase 1 |
| FR-002 | US-005, US-006 | ✓ Done |
| FR-003 | US-007 | Phase 1 |
| FR-004 | US-008 | ✓ Done |
| FR-005 | US-010 | Phase 2 |
| FR-006 | US-009 | Phase 2 |
| FR-007 | US-011, US-012 | Phase 3 |
