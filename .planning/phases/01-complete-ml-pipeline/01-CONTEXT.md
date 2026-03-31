# Phase 1: Complete ML Pipeline - Context

**Gathered:** 2025-01-15
**Status:** Ready for planning
**Source:** PRD Express Path (prd.json - US-004, US-007)

<domain>
## Phase Boundary

This phase completes the core ML pipeline by implementing:
1. **SciBERT Classifier** - Fine-tuned on Papers With Code for reproducibility classification
2. **SHAP Explainer** - Sentence-level attribution showing which text influenced the score

After this phase, all 4 core ML modules will integrate: Classifier + Gaps + Hints + Explain.

</domain>

<decisions>
## Implementation Decisions

### SciBERT Classifier (US-004)
- **Model:** allenai/scibert_scivocab_uncased from HuggingFace (locked)
- **Training:** HuggingFace Trainer API, 5 epochs, lr=2e-5 (locked)
- **Long text handling:** Sliding window - 512 tokens, 64 overlap (locked)
- **Output:** score (0-1), label, confidence, logits (locked)
- **Logging:** Weights & Biases integration (locked)
- **Target:** AUROC > 0.78 (must beat baseline by >5 points) (locked)
- **Checkpoint path:** models/scibert_finetuned/ (locked)

### SHAP Explainer (US-007)
- **Method:** Ablation-based SHAP approximation (locked)
- **Granularity:** Sentence-level attribution via masking (locked)
- **Score range:** Normalized to [-1, 1] (locked)
- **Output:** Top 5 most influential sentences with rank and score (locked)
- **Highlighting:** Color coding - green (positive), yellow (negative), none (neutral) (locked)
- **Caching:** Hash-based result caching to avoid re-computation (locked)
- **Performance target:** ~6s for 20-sentence text on CPU (locked)

### Agent's Discretion
- Exact sliding window implementation details
- SHAP approximation algorithm specifics (leave-one-out vs other)
- Caching strategy (file vs memory)
- W&B project naming and run configuration
- How to handle GPU vs CPU training

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Modules
- `src/classifier.py` — Contains BaselineClassifier (working) and ReproducibilityClassifier (scaffold to complete)
- `src/gap_detector.py` — Gap detection module (reference for integration pattern)
- `src/hint_generator.py` — Hint generation module (reference for integration pattern)

### Data Files
- `data/processed/train.parquet` — 22 training samples (small, will overfit)
- `data/processed/val.parquet` — 4 validation samples
- `data/processed/test.parquet` — 4 test samples
- `data/checklist/neurips_checklist.json` — 43-item checklist

### Tests
- `tests/test_integration.py` — 6 existing tests, extend for SHAP

### PRD
- `prd.json` — US-004 and US-007 acceptance criteria

</canonical_refs>

<specifics>
## Specific Ideas

### SciBERT Training
- Use existing `ReproducibilityClassifier` scaffold in `src/classifier.py`
- Dataset is tiny (22 train) - expect overfitting, that's OK for MVP
- Consider Google Colab if local GPU unavailable
- W&B can be optional (check for WANDB_API_KEY)

### SHAP Implementation
- Create new `src/explainer.py` module
- Leave-one-out sentence masking is simplest approach
- Cache key = hash(text + model_checkpoint)
- Integrate with existing classifier's `predict()` method

### Integration
- Add `explain()` method that returns both classification and SHAP
- Update integration tests to include SHAP

</specifics>

<deferred>
## Deferred Ideas

- Full Papers With Code dataset (using 30-paper sample for now)
- Advanced sliding window strategies (mean vs max pooling)
- GPU-optimized batched SHAP computation
- SHAP visualization library integration

</deferred>

---

*Phase: 01-complete-ml-pipeline*
*Context gathered: 2025-01-15 via PRD Express Path*
