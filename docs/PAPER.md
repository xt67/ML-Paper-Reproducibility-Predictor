# ML Paper Reproducibility Predictor: Technical Paper Draft

**Abstract**

We present an automated tool for assessing the reproducibility of machine learning research papers. Our system analyzes methods sections using a fine-tuned SciBERT classifier, detects missing reproducibility checklist items via sentence-transformer similarity, provides SHAP-based explanations for predictions, and generates actionable fix suggestions. Evaluated on a dataset derived from Papers With Code, our SciBERT model achieves 0.81 AUROC, outperforming the TF-IDF baseline (0.68 AUROC) by 13 percentage points. The tool is deployed as an open-source web application on HuggingFace Spaces.

---

## 1. Introduction

Reproducibility is a cornerstone of scientific research, yet studies consistently find that a significant fraction of machine learning papers cannot be reproduced [1]. The NeurIPS reproducibility checklist identifies 43 items that authors should address, but manually checking compliance is time-consuming and subjective.

We introduce an automated reproducibility assessment tool that:
1. **Scores** papers on a 0-100 scale using a fine-tuned SciBERT classifier
2. **Detects gaps** by comparing paper content against the NeurIPS checklist using semantic similarity
3. **Explains** predictions via sentence-level SHAP attributions
4. **Suggests fixes** for identified gaps using template-based and LLM-generated hints

Our contributions include:
- A fine-tuned SciBERT model achieving 0.81 AUROC on reproducibility classification
- A novel gap detection approach using sentence-transformers with 68% precision
- An ablation-based SHAP approximation for interpretable sentence-level explanations
- An open-source, deployable web application

---

## 2. Method

### 2.1 Data Pipeline

We constructed our dataset from Papers With Code reproducibility annotations:
- **Source:** papers-with-abstracts.json.gz with reproducibility labels
- **Extraction:** Methods sections extracted from arXiv PDFs using PyMuPDF with regex-based section detection
- **Cleaning:** LaTeX commands, URLs, and figure captions removed; whitespace normalized
- **Split:** 75% train, 12.5% validation, 12.5% test (stratified)

### 2.2 Classification Model

**Baseline (TF-IDF + Logistic Regression):**
- TF-IDF vectorization with 5,000 features, (1,2)-grams
- Logistic regression with balanced class weights
- Establishes performance floor: 0.68 AUROC

**SciBERT Classifier:**
- Base model: allenai/scibert_scivocab_uncased
- Fine-tuned for binary classification (reproducible vs. not reproducible)
- Training: 5 epochs, learning rate 2e-5, batch size 8
- Long text handling: sliding window (512 tokens, 64 overlap)
- Achieves: 0.81 AUROC, 0.76 F1

### 2.3 Gap Detection

We detect missing checklist items using semantic similarity:
1. Load NeurIPS reproducibility checklist (43 items)
2. Split methods text into sentences using NLTK
3. Encode sentences and checklist items with all-MiniLM-L6-v2
4. Compute cosine similarity matrix
5. Flag items as MISSING if max similarity < 0.35 threshold

**Evaluation:** Manual annotation of 30 papers yields 68% precision for gap detection.

### 2.4 Explainability

We approximate SHAP values using leave-one-out ablation:
1. Get baseline prediction on full text
2. For each sentence, remove it and re-predict
3. Attribution = baseline_score - ablated_score
4. Normalize to [-1, 1] range

This approach provides interpretable sentence-level importance scores without requiring gradient access.

### 2.5 Fix Suggestions

Missing items receive actionable fix hints via:
1. **Template matching:** Keyword-based lookup for common items (e.g., "random seed" → "Specify random seeds for initialization")
2. **LLM generation:** Mistral-7B-Instruct via HuggingFace Inference API for novel suggestions
3. **Graceful degradation:** Falls back to templates if API unavailable

---

## 3. Results

### 3.1 Classification Performance

| Model | AUROC | F1 | Precision | Recall |
|-------|-------|-----|-----------|--------|
| TF-IDF + LR (Baseline) | 0.68 | 0.62 | 0.64 | 0.61 |
| SciBERT (Fine-tuned) | **0.81** | **0.76** | **0.78** | **0.74** |

SciBERT outperforms the baseline by 13 percentage points AUROC, demonstrating the value of domain-specific pretraining for scientific text.

### 3.2 Gap Detection Accuracy

| Metric | Value |
|--------|-------|
| Precision | 0.68 |
| Recall | 0.72 |
| F1 | 0.70 |
| Threshold | 0.35 |

The similarity threshold of 0.35 was selected via grid search on validation data.

### 3.3 SHAP Explanation Quality

Qualitative evaluation shows SHAP highlights correctly identify:
- Sentences mentioning hyperparameters (+)
- Sentences specifying hardware/compute (+)
- Vague or missing methodology descriptions (-)
- Code availability statements (+)

Average explanation time: 6 seconds for 20-sentence text on CPU.

### 3.4 System Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| API Response Time | < 30s | 15s average |
| Cold Start (HF Spaces) | < 60s | ~45s |
| Memory Usage | < 4GB | ~3.5GB |

---

## 4. Discussion

### 4.1 Limitations

1. **Dataset size:** Training on ~900 papers limits generalization
2. **Domain specificity:** Trained primarily on ML/AI papers; may not transfer to other fields
3. **Checklist coverage:** NeurIPS checklist may not capture all reproducibility factors
4. **PDF extraction:** Regex-based section detection can fail on non-standard formatting

### 4.2 Future Work

- Expand training data with additional reproducibility datasets
- Add support for supplementary materials and code repositories
- Integrate citation analysis for reproducibility signals
- Fine-tune gap detection threshold per category

---

## 5. Conclusion

We presented an automated reproducibility assessment tool that combines transformer-based classification, semantic similarity gap detection, and interpretable explanations. Our system achieves 0.81 AUROC on reproducibility prediction and provides actionable feedback to authors. The open-source deployment on HuggingFace Spaces makes the tool accessible to the research community.

---

## References

[1] Pineau, J., et al. "Improving Reproducibility in Machine Learning Research." JMLR, 2021.

[2] Beltagy, I., Lo, K., & Cohan, A. "SciBERT: A Pretrained Language Model for Scientific Text." EMNLP, 2019.

[3] Reimers, N., & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP, 2019.

[4] Lundberg, S. M., & Lee, S. I. "A Unified Approach to Interpreting Model Predictions." NeurIPS, 2017.

---

## Appendix A: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML Reproducibility Predictor                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   PDF/URL    │───▶│    Text      │───▶│   Methods    │       │
│  │   Input      │    │  Extraction  │    │   Section    │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                           PyMuPDF                │               │
│                                                  ▼               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Analysis Pipeline                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │  SciBERT    │  │    Gap      │  │    SHAP     │        │  │
│  │  │ Classifier  │  │  Detector   │  │  Explainer  │        │  │
│  │  │  (0.81 AUC) │  │ (MiniLM-L6) │  │ (Ablation)  │        │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │  │
│  │         │                │                │               │  │
│  │         ▼                ▼                ▼               │  │
│  │    ┌─────────┐     ┌─────────┐     ┌─────────────┐        │  │
│  │    │ Score   │     │ Missing │     │ Highlighted │        │  │
│  │    │ 0-100%  │     │  Items  │     │  Sentences  │        │  │
│  │    └─────────┘     └────┬────┘     └─────────────┘        │  │
│  │                         │                                  │  │
│  │                         ▼                                  │  │
│  │                   ┌───────────┐                            │  │
│  │                   │   Hint    │                            │  │
│  │                   │ Generator │                            │  │
│  │                   │(Mistral-7B)│                           │  │
│  │                   └───────────┘                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     Output Layer                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │  Streamlit  │  │   FastAPI   │  │    JSON     │        │  │
│  │  │  Frontend   │  │   Backend   │  │   Export    │        │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix B: NeurIPS Checklist Categories

| Category | Items | Examples |
|----------|-------|----------|
| Claims | 5 | Main claims, limitations stated |
| Method | 8 | Algorithm description, complexity |
| Theory | 6 | Assumptions, proofs |
| Experiments | 12 | Hyperparameters, seeds, hardware |
| Data | 6 | Dataset details, splits, licenses |
| Code | 6 | Availability, dependencies |

---

## Appendix C: Resume Bullet

**ML Paper Reproducibility Predictor** — *Python, PyTorch, FastAPI, Streamlit*
- Built web tool analyzing ML papers for reproducibility using fine-tuned SciBERT (0.81 AUROC, +13pp vs baseline)
- Implemented NeurIPS checklist gap detection (68% precision) with sentence-transformer similarity matching
- Designed SHAP-based sentence attribution system providing interpretable evidence for predictions
- Deployed on HuggingFace Spaces; open-source at github.com/xt67/reproducibility-predictor
