# PRD — ML Paper Reproducibility Predictor ## A Claude Code Project Document (Full Specification)
**Owner:** Rayan Rahman
 **Version:** 1.0
 **Status:** Ready to build
 **Target:** 6 weeks, solo, ~8–10h/week
 **Goal:** Live public demo on HuggingFace Spaces + resume bullet + publication draft
---
## 1. Project Overview
### 1.1 What This Is
A web tool that takes the **methods section of any ML/AI research paper** and outputs: 1. A **reproducibility score** (0–100) — how likely is this paper to be reproducible? 2. A **gap report** — which specific reporting items are missing (e.g. no random seed, no dataset split info)? 3. **Evidence highlights** — which sentences drove the score, with SHAP-based attribution? 4. **Fix hints** — one-line suggestions for each detected gap?
### 1.2 Why It Exists
The ML reproducibility crisis is well-documented. Over 70% of ML papers cannot be reproduced by independent researchers. The root cause is not fraud — it is **incomplete reporting**. Authors forget to mention random seeds, hardware specs, dataset split ratios, hyperparameter search ranges, and so on.
Existing tools (Papers With Code badges, NeurIPS checklist) are manual. No automated, explainable, publicly accessible scorer exists. This is the gap this project fills.
### 1.3 Who Uses It
- **Researchers** submitting papers who want to self-check before submission - **Reviewers** doing a quick reproducibility pre-screen - **Conference organizers** who want to batch-score submissions - **Students** (like yourself) building their portfolio with something genuinely useful
### 1.4 What Makes It Novel
The key technical contribution beyond simple classification: - **SHAP-based sentence-level attribution** on a SciBERT model — showing *why* a paper scored low, not just *that* it scored low - **Checklist-grounded gap detection** using sentence-transformer similarity against the NeurIPS reproducibility checklist - **LLM-generated fix hints** per gap — turning detection into actionable repair suggestions
This combination (classify + explain + detect gaps + suggest fixes) does not exist as a single open tool.
---
## 2. Tech Stack
``` Backend



FastAPI (Python 3.10+) ML Model


 HuggingFace Transformers — allenai/scibert_scivocab_uncased Explainability SHAP (shap library) Gap Detection
sentence-transformers — all-MiniLM-L6-v2 Fix Hints


HuggingFace Inference API — mistralai/Mistral-7B-Instruct-v0.2 (free tier) PDF Parsing

pdfminer.six + PyMuPDF (fitz) Frontend


 Streamlit Experiment Log Weights & Biases (free tier) Deployment

 HuggingFace Spaces (free, Streamlit SDK) Version Ctrl
 Git + GitHub (public repo) Data Storage
 Parquet files (no database needed) ```
No paid services required. Everything runs free.
---
## 3. Folder Structure
Create this exact structure at project start. Do not deviate.
``` reproducibility-predictor/ │ ├── data/ │
 ├── raw/











# Downloaded CSVs from Papers With Code │
 ├── processed/ │
 │
 ├── train.parquet │
 │
 ├── val.parquet │
 │
 └── test.parquet │
 └── checklist/ │


 └── neurips_checklist.json
# 43 checklist items as JSON list │ ├── models/ │
 └── scibert_finetuned/




# Saved HuggingFace model checkpoint │


 ├── config.json │


 ├── pytorch_model.bin │


 └── tokenizer/ │ ├── src/ │
 ├── __init__.py │
 ├── data_pipeline.py





# Download, clean, split data │
 ├── pdf_extractor.py





# Extract methods section from PDF │
 ├── classifier.py






 # SciBERT fine-tuning + inference │
 ├── gap_detector.py





 # Checklist-based gap detection │
 ├── explainer.py







# SHAP sentence attribution │
 ├── hint_generator.py




 # LLM fix hints via HF Inference API │
 └── pipeline.py







 # Orchestrator — calls all modules in order │ ├── api/ │
 ├── __init__.py │
 ├── main.py









 # FastAPI app entry point │
 ├── models.py








 # Pydantic request/response models │
 └── routers/ │


 ├── __init__.py │


 ├── analyze.py






# POST /analyze endpoint │


 └── health.py






 # GET /health endpoint │ ├── app/ │
 └── streamlit_app.py





# Streamlit frontend │ ├── notebooks/ │
 ├── 01_data_exploration.ipynb │
 ├── 02_baseline_experiments.ipynb │
 └── 03_shap_analysis.ipynb │ ├── tests/ │
 ├── test_classifier.py │
 ├── test_gap_detector.py │
 └── test_pipeline.py │ ├── scripts/ │
 ├── download_data.py





# One-shot data download script │
 ├── train_model.py






# Training entry point │
 └── evaluate_model.py




# Eval entry point │ ├── requirements.txt ├── requirements-dev.txt ├── .env.example









# Template for secrets (HF token etc.) ├── .gitignore ├── README.md └── Dockerfile










# For HF Spaces deployment ```
---
## 4. Data Pipeline (Week 1)
### 4.1 Dataset Source
**Primary:** Papers With Code Reproducibility Dataset - URL: https://paperswithcode.com/datasets (search "reproducibility") - Alternatively: https://github.com/paperswithcode/paperswithcode-data - Download `papers-with-abstracts.json.gz` and the reproducibility annotation file - You need: `paper_id`, `arxiv_id`, `reproducibility_score` (binary: 0 or 1), `title`
**Secondary (for methods text):** arXiv API - Use the `arxiv` Python library to fetch each paper by arxiv_id - Download the PDF, extract methods section using `pdf_extractor.py`
**Checklist:** NeurIPS Reproducibility Checklist - Source: https://neurips.cc/public/guides/PaperChecklist - Manually transcribe the 43 items into `data/checklist/neurips_checklist.json` - Format: `[{"id": 1, "item": "Includes a conceptual outline and/or pseudocode...", "severity": "high"}, ...]` - Severity mapping: hyperparameters/seeds/splits = high, hardware/runtime = medium, limitations = low
### 4.2 data_pipeline.py — Full Specification
```python # src/data_pipeline.py # Responsibilities: # 1. Load raw Papers With Code annotations # 2. For each paper, fetch PDF from arXiv and extract methods section # 3. Clean text # 4. Split into train/val/test # 5. Save as Parquet
# Functions to implement:
def load_pwc_annotations(filepath: str) -> pd.DataFrame:

 """

 Load the PwC reproducibility CSV.

 Returns DataFrame with columns: paper_id, arxiv_id, label (0 or 1), title

 Expected rows: ~1200

 """
def fetch_arxiv_pdf(arxiv_id: str, save_dir: str) -> str:

 """

 Download PDF from arxiv.org/pdf/{arxiv_id}.pdf

 Save to save_dir/{arxiv_id}.pdf

 Return local filepath.

 Use requests with a 10s timeout. Retry 3 times on failure.

 Sleep 1s between requests to avoid rate limiting.

 """
def extract_methods_section(pdf_path: str) -> str:

 """

 Extract the methods/methodology section from a PDF.

 Strategy:

 1. Use pdfminer.six to extract full text page by page

 2. Search for section headers matching regex:



r'(methods?|methodology|approach|proposed method|our method|model|architecture)'



(case-insensitive, at start of line or after newline)

 3. Extract text from that header until the next major section header



(results, experiments, evaluation, conclusion, related work)

 4. If no methods section found, fall back to abstract + first 1500 chars of body

 5. Strip references, figure captions, table captions

 Return: cleaned string, max 2000 tokens (truncate if longer)

 """
def clean_text(text: str) -> str:

 """

 - Remove LaTeX commands: \begin{}, \end{}, \cite{}, \ref{}, $$...$$

 - Remove URLs

 - Normalize whitespace (no double spaces, no triple newlines)

 - Remove lines that are purely numbers (page numbers, table data)

 - Strip leading/trailing whitespace

 Return cleaned string.

 """
def split_dataset(df: pd.DataFrame, train=0.75, val=0.125, test=0.125, seed=42):

 """

 Stratified split on the label column (maintain class balance in each split).

 Save each split as Parquet to data/processed/

 Print split sizes and class distributions.

 """
def run_pipeline():

 """

 Main function. Runs all steps above in order.

 Logs progress to console with tqdm progress bars.

 Skips PDFs that already exist on disk (resumable).

 Skips papers where extraction fails (log to failed_extractions.txt).

 """ ```
### 4.3 Expected Data Stats After Pipeline Runs
| Split | Rows | Label=1 (reproducible) | Label=0 | |-------|------|------------------------|---------| | Train | ~900 | ~450 | ~450 | | Val
 | ~150 | ~75
| ~75
| | Test
| ~150 | ~75
| ~75
|
If real class balance differs, use class weights in training — do NOT oversample.
---
## 5. PDF Extractor (Week 1)
### 5.1 pdf_extractor.py — Full Specification
```python # src/pdf_extractor.py
import fitz
# PyMuPDF import re
SECTION_HEADERS = [

 r'\b(methods?|methodology|approach|proposed method)\b',

 r'\b(model architecture|our model|our approach)\b',

 r'\b(system design|technical approach)\b', ]
END_MARKERS = [

 r'\b(results?|experiments?|evaluation|empirical)\b',

 r'\b(conclusion|related work|discussion|limitations?)\b',

 r'\b(references?|bibliography|appendix)\b', ]
def extract_text_from_pdf(pdf_path: str) -> str:

 """

 Use PyMuPDF to extract full text, preserving paragraph structure.

 Join pages with double newline.

 """
def find_methods_section(full_text: str) -> str:

 """

 Find methods section using SECTION_HEADERS and END_MARKERS regex.

 Return the slice of text between them.

 If not found: return first 2000 characters of body text (skip abstract).

 """
def extract_from_url(arxiv_url: str) -> str:

 """

 Download PDF from URL to a temp file, extract, delete temp file.

 Used by the API for live URL input.

 """ ```
---
## 6. Classifier Module (Week 2)
### 6.1 Baseline First
Before fine-tuning SciBERT, always implement the baseline:
```python # In notebooks/02_baseline_experiments.ipynb
from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.linear_model import LogisticRegression from sklearn.metrics import roc_auc_score, classification_report
# TF-IDF + LR baseline # Expected AUROC: 0.65–0.72 # This is your "floor" — SciBERT must beat this ```
Log baseline results. Include in final paper/README as Table 1 row.
### 6.2 classifier.py — Full Specification
```python # src/classifier.py
MODEL_NAME = "allenai/scibert_scivocab_uncased" MAX_LENGTH = 512
# SciBERT max tokens BATCH_SIZE = 8

# Adjust down to 4 if GPU OOM LEARNING_RATE = 2e-5 EPOCHS = 5 WARMUP_RATIO = 0.1 WEIGHT_DECAY = 0.01
class ReproducibilityClassifier:

 """

 Wraps HuggingFace SciBERT for binary classification.

 """


def __init__(self, model_path: str = None):



 """



 If model_path is None, load pretrained SciBERT.



 If model_path given, load fine-tuned checkpoint.



 """


def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str):



 """



 Fine-tune on train_df, evaluate on val_df after each epoch.








Training setup:



 - Use HuggingFace Trainer API



 - TrainingArguments:





 output_dir=output_dir,





 num_train_epochs=EPOCHS,





 per_device_train_batch_size=BATCH_SIZE,





 per_device_eval_batch_size=BATCH_SIZE,





 warmup_ratio=WARMUP_RATIO,





 weight_decay=WEIGHT_DECAY,





 evaluation_strategy="epoch",





 save_strategy="best",





 load_best_model_at_end=True,





 metric_for_best_model="eval_auroc",





 logging_dir="./logs",





 report_to="wandb"
# Free W&B logging








- Compute metrics function: calculate AUROC, F1, precision, recall



 - Use class weights if class imbalance > 1.5:1



 - Save best checkpoint to models/scibert_finetuned/








Print final eval metrics table at end.



 """


def predict(self, text: str) -> dict:



 """



 Input: raw methods section text (string)



 Output: {





 "score": float,



# 0.0 to 1.0 probability of being reproducible





 "label": int,




# 0 or 1





 "confidence": float,
 # max(prob_0, prob_1)





 "logits": list



 # raw logits [logit_0, logit_1]



 }



 Tokenize, run forward pass, return softmax probabilities.



 Handle texts longer than MAX_LENGTH by sliding window:





 - Split into 512-token windows with 64-token overlap





 - Average logits across windows





 - Apply softmax to averaged logits



 """


def predict_batch(self, texts: list[str]) -> list[dict]:



 """



 Batch prediction for evaluation.



 Process in chunks of BATCH_SIZE.



 Show tqdm progress bar.



 """


def save(self, path: str):



 """Save model + tokenizer to path."""


def load(self, path: str):



 """Load model + tokenizer from path.""" ```
### 6.3 Training Script
```python # scripts/train_model.py # Entry point for training. Run with: python scripts/train_model.py
import wandb wandb.init(project="reproducibility-predictor", name="scibert-v1")
# Load data # Initialize classifier # Train # Save # Print final test set metrics ```
### 6.4 Evaluation Targets
| Metric | Baseline (TF-IDF+LR) | Target (SciBERT) | |--------|----------------------|------------------| | AUROC | 0.65–0.72 | > 0.78 | | F1 (macro) | 0.62 | > 0.74 | | Precision | — | > 0.72 | | Recall | — | > 0.72 |
If SciBERT does not beat baseline by >5 points AUROC, check: 1. Is MAX_LENGTH sufficient? (try 256 vs 512) 2. Is learning rate too high? (try 1e-5) 3. Is the methods extraction working? (manually inspect 10 samples)
---
## 7. Gap Detector Module (Week 3)
### 7.1 NeurIPS Checklist JSON Format
```json [
 {

 "id": 1,

 "category": "claims",

 "item": "All claims in the paper are supported by theoretical or experimental evidence.",

 "severity": "high",

 "keywords": ["claims", "evidence", "theoretical", "experimental"]
 },
 {

 "id": 2,

 "category": "reproducibility",

 "item": "The paper specifies all hyperparameters used, including learning rate, batch size, and optimizer settings.",

 "severity": "high",

 "keywords": ["hyperparameters", "learning rate", "batch size", "optimizer"]
 }
 // ... all 43 items ] ```
Severity levels: - **high** (must-have): random seeds, dataset splits, hyperparameters, model architecture details, dataset name + version - **medium** (should-have): hardware specs, training time, number of runs, validation procedure - **low** (nice-to-have): limitations section, broader impact, code availability
### 7.2 gap_detector.py — Full Specification
```python # src/gap_detector.py
from sentence_transformers import SentenceTransformer, util import json
SIMILARITY_THRESHOLD = 0.35
# Below this = item is missing SENTENCE_MODEL = "all-MiniLM-L6-v2"
class GapDetector:


def __init__(self, checklist_path: str):



 """



 Load checklist JSON.



 Load sentence transformer model.



 Pre-encode all checklist items (do this once at init, cache embeddings).



 """


def detect(self, methods_text: str) -> list[dict]:



 """



 Input: methods section text








Steps:



 1. Split methods_text into sentences using nltk.sent_tokenize()



 2. Encode all sentences using sentence transformer



 3. For each checklist item:





a. Compute cosine similarity against all sentence embeddings





b. max_sim = max similarity across all sentences





c. If max_sim < SIMILARITY_THRESHOLD: flag as MISSING





d. If max_sim >= THRESHOLD: flag as PRESENT, record best_sentence








Output: list of dicts, one per checklist item:



 [




 {





 "id": 1,





 "item": "The paper specifies all hyperparameters...",





 "severity": "high",





 "status": "missing",


 # or "present"





 "similarity_score": 0.21,
# max cosine sim found





 "best_matching_sentence": ""
# empty if missing




 },




 ...



 ]



 """


def summary(self, gaps: list[dict]) -> dict:



 """



 Compute summary stats:



 {





 "total_items": 43,





 "present": 31,





 "missing": 12,





 "missing_high_severity": 4,





 "missing_medium_severity": 5,





 "missing_low_severity": 3,





 "coverage_score": 72.1

# (present/total) * 100



 }



 """ ```
### 7.3 Manual Evaluation of Gap Detector
After building, manually evaluate on 30 papers: 1. Pick 15 papers you know are reproducible (check Papers With Code) 2. Pick 15 papers you know are not 3. For each: run `detect()`, then manually read the paper and check if gaps are real 4. Compute precision = (true missing items flagged) / (all items flagged as missing) 5. Target precision > 0.68 on your manual eval set 6. Document this in README as "Gap Detector Evaluation" section
---
## 8. Explainability Module (Week 4)
### 8.1 explainer.py — Full Specification
```python # src/explainer.py
import shap import numpy as np import nltk
class SHAPExplainer:


def __init__(self, classifier: ReproducibilityClassifier):



 """



 Initialize SHAP explainer.



 Use shap.Explainer with the classifier's predict function.



 The predict function must accept a list of strings and return



 a 2D numpy array of shape (n_samples, 2) — probabilities for each class.



 """


def explain(self, text: str, target_class: int = 1) -> list[dict]:



 """



 Compute SHAP values for the given text.








Steps:



 1. Split text into sentences with nltk.sent_tokenize()



 2. For each sentence, compute its SHAP value contribution to





the target_class prediction by:





a. Getting baseline prediction with full text





b. Getting prediction with sentence masked (replaced by empty string)





c. sentence_shap = baseline_prob - masked_prob





(This is a simple ablation-based approximation — cheaper than full SHAP





 and sufficient for sentence-level attribution)



 3. Normalize SHAP values to [-1, 1] range



 4. Sort sentences by |shap_value| descending








Output: list of dicts (top 5 sentences only):



 [




 {





 "sentence": "We used a learning rate of 0.001...",





 "shap_value": 0.142,


# positive = pushes toward reproducible





 "rank": 1,





 "normalized_score": 0.87
# for display (0–1 scale, 1 = most influential)




 },




 ...



 ]








Note on performance: with 20-sentence methods sections, this runs



 20 forward passes. Each takes ~0.3s on CPU. Total: ~6s. Acceptable.



 Cache results by text hash to avoid re-computation.



 """


def get_highlighted_text(self, text: str, explanations: list[dict]) -> list[dict]:



 """



 Split full text into sentences.



 For each sentence, return:



 {





 "text": sentence_text,





 "highlight_score": float,
# 0.0 to 1.0





 "highlight_color": str

 # "green", "yellow", "none"





 # green = score > 0.6 (strongly pushes toward reproducible)





 # yellow = score 0.3–0.6





 # none = score < 0.3



 }



 This is what the frontend uses to render highlighted text.



 """ ```
---
## 9. Fix Hint Generator (Week 4)
### 9.1 hint_generator.py — Full Specification
```python # src/hint_generator.py
import requests import os
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
class HintGenerator:


def __init__(self):



 """



 Load HF_TOKEN from environment variable.



 If not set, skip hint generation and return empty hints.



 (Tool still works without hints — degrade gracefully)



 """



 self.token = os.getenv("HF_TOKEN", None)


def generate_hint(self, gap_item: str, severity: str) -> str:



 """



 Generate a one-line fix hint for a missing checklist item.








Prompt template:



 <s>[INST] You are reviewing an ML paper's methods section.



 The following required reporting item is missing:



 "{gap_item}"








Write exactly ONE sentence (max 25 words) that the author should add



 to their methods section to address this gap. Be specific and concrete.



 Do not start with "I" or explain what you're doing. Just write the sentence. [/INST]








API call:



 - POST to HF_API_URL



 - headers: {"Authorization": f"Bearer {self.token}"}



 - payload: {"inputs": prompt, "parameters": {"max_new_tokens": 60, "temperature": 0.3}}



 - timeout: 15s



 - If API fails (rate limit, timeout, error): return a template fallback:




 f"Specify {gap_item.lower()[:50]} explicitly in the methods section."








Return: single string, the generated hint sentence



 """


def generate_hints_for_gaps(self, gaps: list[dict]) -> list[dict]:



 """



 Only generate hints for items with status="missing" AND severity in ["high", "medium"].



 Skip severity="low" to save API calls.








Return the input gaps list with "hint" field added to each item:



 - missing + high/medium: hint = generated sentence



 - missing + low: hint = ""



 - present: hint = ""








Rate limit: sleep 0.5s between API calls.



 """ ```
---
## 10. Pipeline Orchestrator (Week 4–5)
### 10.1 pipeline.py — Full Specification
```python # src/pipeline.py # This is the single entry point for all inference. # The API, the Streamlit app, and tests all call ONLY this module.
import hashlib import json from dataclasses import dataclass
@dataclass class AnalysisResult:

 """Structured output of a full analysis run."""




# Input

 input_text: str




 # Raw methods section text

 text_hash: str





# MD5 hash for caching




# Classifier output

 reproducibility_score: float

# 0.0–1.0

 reproducibility_label: str


# "reproducible" or "not reproducible"

 confidence: float






 # 0.0–1.0

 score_out_of_100: int




 # round(score * 100)




# Gap detection

 gaps: list[dict]







# Full gap list from GapDetector

 gap_summary: dict






 # Summary stats




# Explainability

 top_sentences: list[dict]


 # Top 5 SHAP-attributed sentences

 highlighted_text: list[dict]

# Full text with highlight scores




# Hints

 hints_generated: bool




 # Whether hints were generated




# Metadata

 processing_time_seconds: float

 model_version: str = "scibert-v1"
 class ReproducibilityPipeline:


def __init__(self, model_path: str, checklist_path: str):



 """



 Initialize all modules:



 1. ReproducibilityClassifier (load from model_path)



 2. GapDetector (load checklist from checklist_path)



 3. SHAPExplainer (wraps classifier)



 4. HintGenerator








Log initialization time.



 This runs ONCE at API startup — not per request.



 """



 self._cache = {}
# Simple in-memory cache: hash -> AnalysisResult


def analyze(self, text: str, generate_hints: bool = True) -> AnalysisResult:



 """



 Full analysis pipeline for a methods section string.








Steps (in order):



 1. Clean text (call clean_text from data_pipeline)



 2. Compute text hash (MD5)



 3. Check cache — if hash in cache, return cached result immediately



 4. Run classifier.predict(text)



 5. Run gap_detector.detect(text)



 6. Run gap_detector.summary(gaps)



 7. Run explainer.explain(text)



 8. Run explainer.get_highlighted_text(text, explanations)



 9. If generate_hints: run hint_generator.generate_hints_for_gaps(gaps)



 10. Build AnalysisResult dataclass



 11. Store in cache



 12. Return result








Wrap entire pipeline in try/except.



 On any exception: log error, raise HTTPException(500) with message.



 Log total processing time.



 """


def analyze_pdf(self, pdf_path: str, generate_hints: bool = True) -> AnalysisResult:



 """



 Extract methods section from PDF, then call analyze().



 """


def analyze_pdf_url(self, url: str, generate_hints: bool = True) -> AnalysisResult:



 """



 Download PDF from URL to temp file, extract, analyze, delete temp file.



 """ ```
---
## 11. FastAPI Backend (Week 5)
### 11.1 Pydantic Models — api/models.py
```python # api/models.py
from pydantic import BaseModel, Field, validator from typing import Optional
# ---- Request Models ----
class TextAnalysisRequest(BaseModel):

 text: str = Field(



 ...,



 min_length=100,



 max_length=10000,



 description="Methods section text to analyze"

 )

 generate_hints: bool = Field(



 default=True,



 description="Whether to generate LLM fix hints (slower if True)"

 )


@validator('text')

 def text_not_empty(cls, v):



 if len(v.strip()) < 100:





 raise ValueError("Text too short. Paste at least the methods section.")



 return v.strip()
 class URLAnalysisRequest(BaseModel):

 url: str = Field(..., description="ArXiv PDF URL or direct PDF URL")

 generate_hints: bool = True


@validator('url')

 def url_must_be_pdf(cls, v):



 if not (v.endswith('.pdf') or 'arxiv.org' in v):





 raise ValueError("URL must point to a PDF file or arXiv paper")



 return v
 # ---- Response Models ----
class GapItem(BaseModel):

 id: int

 item: str

 severity: str




 # "high", "medium", "low"

 status: str





 # "missing", "present"

 similarity_score: float

 best_matching_sentence: str

 hint: str






 # "" if no hint generated
 class SentenceHighlight(BaseModel):

 text: str

 highlight_score: float

 highlight_color: str

# "green", "yellow", "none"
 class TopSentence(BaseModel):

 sentence: str

 shap_value: float

 rank: int

 normalized_score: float
 class GapSummary(BaseModel):

 total_items: int

 present: int

 missing: int

 missing_high_severity: int

 missing_medium_severity: int

 missing_low_severity: int

 coverage_score: float
 class AnalysisResponse(BaseModel):

 # Score

 reproducibility_score: float

 score_out_of_100: int

 reproducibility_label: str

 confidence: float




# Gaps

 gaps: list[GapItem]

 gap_summary: GapSummary




# Explainability

 top_sentences: list[TopSentence]

 highlighted_text: list[SentenceHighlight]




# Meta

 hints_generated: bool

 processing_time_seconds: float

 model_version: str
 class HealthResponse(BaseModel):

 status: str



 # "ok"

 model_loaded: bool

 version: str ```
### 11.2 Routers
```python # api/routers/health.py
from fastapi import APIRouter from api.models import HealthResponse
router = APIRouter()
@router.get("/health", response_model=HealthResponse, tags=["Health"]) async def health_check():

 """

 Returns API health status.

 Used by HuggingFace Spaces to check if app is ready.

 """

 return HealthResponse(



 status="ok",



 model_loaded=True,
# Will be False if pipeline init failed



 version="1.0.0"

 ) ```
```python # api/routers/analyze.py
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks from api.models import TextAnalysisRequest, URLAnalysisRequest, AnalysisResponse import tempfile import os
router = APIRouter()
# Pipeline instance will be injected via app.state in main.py
@router.post("/analyze/text", response_model=AnalysisResponse, tags=["Analysis"]) async def analyze_text(request: TextAnalysisRequest):

 """

 Analyze a methods section provided as raw text.




- Input: JSON body with 'text' field (100–10000 chars)

 - Output: Full analysis result

 - Typical latency: 5–15 seconds depending on text length

 - Cached: Yes (same text hash returns instantly)

 """

 try:



 result = pipeline.analyze(request.text, request.generate_hints)



 return _result_to_response(result)

 except Exception as e:



 raise HTTPException(status_code=500, detail=str(e))
 @router.post("/analyze/url", response_model=AnalysisResponse, tags=["Analysis"]) async def analyze_url(request: URLAnalysisRequest):

 """

 Analyze a paper given its PDF URL (arXiv or direct PDF link).

 Downloads the PDF, extracts methods section, runs analysis.




- Latency: 10–30 seconds (includes PDF download + extraction)

 """

 try:



 result = pipeline.analyze_pdf_url(request.url, request.generate_hints)



 return _result_to_response(result)

 except Exception as e:



 raise HTTPException(status_code=500, detail=str(e))
 @router.post("/analyze/upload", response_model=AnalysisResponse, tags=["Analysis"]) async def analyze_upload(file: UploadFile = File(...), generate_hints: bool = True):

 """

 Analyze a paper from an uploaded PDF file.

 Max file size: 10MB.

 """

 if file.content_type != "application/pdf":



 raise HTTPException(status_code=400, detail="Only PDF files accepted")




# Save to temp file

 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:



 content = await file.read()



 if len(content) > 10 * 1024 * 1024:
# 10MB limit





 raise HTTPException(status_code=413, detail="File too large (max 10MB)")



 tmp.write(content)



 tmp_path = tmp.name




try:



 result = pipeline.analyze_pdf(tmp_path, generate_hints)



 return _result_to_response(result)

 except Exception as e:



 raise HTTPException(status_code=500, detail=str(e))

 finally:



 os.unlink(tmp_path)
# Always delete temp file
 def _result_to_response(result) -> AnalysisResponse:

 """Convert AnalysisResult dataclass to AnalysisResponse Pydantic model."""

 return AnalysisResponse(



 reproducibility_score=result.reproducibility_score,



 score_out_of_100=result.score_out_of_100,



 reproducibility_label=result.reproducibility_label,



 confidence=result.confidence,



 gaps=[GapItem(**g) for g in result.gaps],



 gap_summary=GapSummary(**result.gap_summary),



 top_sentences=[TopSentence(**s) for s in result.top_sentences],



 highlighted_text=[SentenceHighlight(**h) for h in result.highlighted_text],



 hints_generated=result.hints_generated,



 processing_time_seconds=result.processing_time_seconds,



 model_version=result.model_version

 ) ```
### 11.3 Main App — api/main.py
```python # api/main.py
from fastapi import FastAPI from fastapi.middleware.cors import CORSMiddleware from contextlib import asynccontextmanager from api.routers import analyze, health from src.pipeline import ReproducibilityPipeline import logging
logging.basicConfig(level=logging.INFO) logger = logging.getLogger(__name__)
MODEL_PATH = "models/scibert_finetuned" CHECKLIST_PATH = "data/checklist/neurips_checklist.json"
# Global pipeline instance pipeline = None
@asynccontextmanager async def lifespan(app: FastAPI):

 """Load pipeline on startup, clean up on shutdown."""

 global pipeline

 logger.info("Loading pipeline...")

 pipeline = ReproducibilityPipeline(MODEL_PATH, CHECKLIST_PATH)

 logger.info("Pipeline ready.")

 yield

 logger.info("Shutting down.")
app = FastAPI(

 title="ML Paper Reproducibility Predictor API",

 description="""

 Analyzes ML paper methods sections and predicts reproducibility.

 Built by Rayan Rahman — github.com/xt67

 """,

 version="1.0.0",

 lifespan=lifespan )
# CORS — allow Streamlit frontend to call API app.add_middleware(

 CORSMiddleware,

 allow_origins=["*"],
# Restrict in production

 allow_methods=["GET", "POST"],

 allow_headers=["*"], )
# Register routers app.include_router(health.router) app.include_router(analyze.router, prefix="/api/v1")
# Inject pipeline into routers @app.on_event("startup") async def inject_pipeline():

 analyze.pipeline = pipeline
if __name__ == "__main__":

 import uvicorn

 uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) ```
### 11.4 Running the API
```bash # Development (with auto-reload) uvicorn api.main:app --reload --port 8000
# Interactive docs auto-generated at: # http://localhost:8000/docs


 (Swagger UI) # http://localhost:8000/redoc


(ReDoc)
# Test with curl: curl -X POST http://localhost:8000/api/v1/analyze/text \
 -H "Content-Type: application/json" \
 -d '{"text": "We trained our model using SGD optimizer with learning rate 0.01 for 100 epochs on the CIFAR-10 dataset...", "generate_hints": false}' ```
---
## 12. Streamlit Frontend (Week 5)
### 12.1 app/streamlit_app.py — Full Specification
```python # app/streamlit_app.py
import streamlit as st import requests import json
API_BASE = "http://localhost:8000/api/v1"
# Change to HF Spaces URL after deploy
st.set_page_config(

 page_title="ML Reproducibility Predictor",

 page_icon="🔬",

 layout="wide" )
# ---- HEADER ---- st.title("ML Paper Reproducibility Predictor") st.markdown(

 "Paste your **methods section** below. "

 "Get a reproducibility score, gap report, and fix suggestions." )
# ---- INPUT SECTION ---- tab1, tab2, tab3 = st.tabs(["Paste text", "ArXiv URL", "Upload PDF"])
with tab1:

 text_input = st.text_area(



 "Methods section text",



 height=250,



 placeholder="Paste the methods/methodology section of your paper here..."

 )

 generate_hints = st.checkbox("Generate fix hints (slower, requires HF API)", value=True)

 analyze_btn = st.button("Analyze", type="primary", key="text_btn")
with tab2:

 url_input = st.text_input(



 "ArXiv URL",



 placeholder="https://arxiv.org/pdf/2301.00001.pdf"

 )

 analyze_url_btn = st.button("Analyze", type="primary", key="url_btn")
with tab3:

 uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

 analyze_upload_btn = st.button("Analyze", type="primary", key="upload_btn")
# ---- EXAMPLE PAPERS ---- with st.expander("Try example papers"):

 col1, col2, col3 = st.columns(3)

 # Pre-load 3 example methods sections (one high score, one low, one medium)

 # Store as constants in the file
# ---- ANALYSIS LOGIC ---- result = None
if analyze_btn and text_input:

 with st.spinner("Analyzing..."):



 resp = requests.post(





 f"{API_BASE}/analyze/text",





 json={"text": text_input, "generate_hints": generate_hints}



 )



 if resp.status_code == 200:





 result = resp.json()



 else:





 st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
# ---- RESULTS DISPLAY ---- if result:

 st.divider()




# --- SCORE SECTION ---

 col1, col2, col3, col4 = st.columns(4)




score = result["score_out_of_100"]

 score_color = "green" if score >= 70 else "orange" if score >= 45 else "red"




with col1:



 st.metric("Reproducibility Score", f"{score}/100")

 with col2:



 st.metric("Verdict", result["reproducibility_label"].title())

 with col3:



 st.metric("Confidence", f"{result['confidence']*100:.0f}%")

 with col4:



 st.metric("Gaps Found", result["gap_summary"]["missing"])




# Score bar

 st.progress(score / 100)




# --- TWO COLUMN LAYOUT ---

 left, right = st.columns([1, 1])




with left:



 # --- GAP REPORT ---



 st.subheader("Gap Report")








gaps = result["gaps"]



 missing_gaps = [g for g in gaps if g["status"] == "missing"]








# Group by severity



 for severity in ["high", "medium", "low"]:





 severity_gaps = [g for g in missing_gaps if g["severity"] == severity]





 if severity_gaps:







 severity_color = {"high": "🔴", "medium": "🟡", "low": "⚪"}[severity]







 st.markdown(f"**{severity_color} {severity.title()} priority** — {len(severity_gaps)} missing")
















for gap in severity_gaps:









 with st.expander(gap["item"][:80] + "..."):











 st.write(f"**Similarity score:** {gap['similarity_score']:.2f}")











 if gap["hint"]:













 st.info(f"**Suggested fix:** {gap['hint']}")




with right:



 # --- EVIDENCE HIGHLIGHTS ---



 st.subheader("Key Sentences")



 st.caption("Sentences that most influenced the score")








for sent in result["top_sentences"]:





 color = {"green": "#d4edda", "yellow": "#fff3cd"}.get(







 "green" if sent["normalized_score"] > 0.6 else "yellow", "#f8f9fa"





 )





 st.markdown(







 f'<div style="background:{color};padding:8px;border-radius:4px;'







 f'margin:4px 0;font-size:13px">{sent["sentence"]}</div>',







 unsafe_allow_html=True





 )




# --- HIGHLIGHTED FULL TEXT ---

 st.subheader("Annotated Methods Section")

 st.caption("Green = strongly supports reproducibility | Yellow = moderate | White = neutral")




html_parts = []

 for part in result["highlighted_text"]:



 bg = {"green": "#c3e6cb", "yellow": "#ffeeba", "none": "transparent"}[part["highlight_color"]]



 html_parts.append(





 f'<span style="background:{bg};padding:2px 1px;border-radius:2px">{part["text"]} </span>'



 )




st.markdown(



 f'<div style="line-height:1.8;font-size:14px">{"".join(html_parts)}</div>',



 unsafe_allow_html=True

 )




# --- DOWNLOAD REPORT ---

 st.divider()

 report = {



 "score": result["score_out_of_100"],



 "verdict": result["reproducibility_label"],



 "gaps": [g for g in result["gaps"] if g["status"] == "missing"],



 "top_sentences": result["top_sentences"]

 }

 st.download_button(



 "Download report (JSON)",



 data=json.dumps(report, indent=2),



 file_name="reproducibility_report.json",



 mime="application/json"

 ) ```
---
## 13. Requirements Files
### requirements.txt
``` fastapi==0.110.0 uvicorn[standard]==0.29.0 pydantic==2.6.4 transformers==4.39.3 torch==2.2.2 sentence-transformers==2.7.0 shap==0.45.0 scikit-learn==1.4.1 pandas==2.2.1 pyarrow==15.0.2 pdfminer.six==20221105 PyMuPDF==1.24.0 arxiv==2.1.0 nltk==3.8.1 requests==2.31.0 python-multipart==0.0.9 tqdm==4.66.2 wandb==0.16.6 streamlit==1.33.0 python-dotenv==1.0.1 ```
### requirements-dev.txt
``` pytest==8.1.1 pytest-asyncio==0.23.6 httpx==0.27.0
 # For FastAPI test client black==24.3.0 flake8==7.0.0 ipykernel==6.29.4 jupyter==1.0.0 ```
---
## 14. Environment Variables
```bash # .env.example
(copy to .env, never commit .env) HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
 # Get from huggingface.co/settings/tokens WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx # Get from wandb.ai/settings MODEL_PATH=models/scibert_finetuned CHECKLIST_PATH=data/checklist/neurips_checklist.json API_BASE_URL=http://localhost:8000 ```
---
## 15. Testing
### 15.1 tests/test_classifier.py
```python import pytest from src.classifier import ReproducibilityClassifier
@pytest.fixture def classifier():

 return ReproducibilityClassifier(model_path="models/scibert_finetuned")
def test_predict_returns_dict(classifier):

 result = classifier.predict("We trained using Adam optimizer with lr=0.001.")

 assert "score" in result

 assert "label" in result

 assert 0.0 <= result["score"] <= 1.0

 assert result["label"] in [0, 1]
def test_predict_high_reproducible(classifier):

 text = """We used the Adam optimizer with a learning rate of 0.001 and beta values


of 0.9 and 0.999. We trained for 100 epochs with batch size 32. The random seed


was set to 42. We used the CIFAR-10 dataset with an 80/10/10 train/val/test split.


All experiments were run on a single NVIDIA A100 GPU."""

 result = classifier.predict(text)

 assert result["score"] > 0.6
# Should predict reproducible
def test_predict_low_reproducible(classifier):

 text = "We trained a neural network on our dataset and achieved state-of-the-art results."

 result = classifier.predict(text)

 assert result["score"] < 0.5
# Should predict not reproducible
def test_long_text_handled(classifier):

 # 600-token text (exceeds MAX_LENGTH=512)

 long_text = "We used Adam optimizer. " * 100

 result = classifier.predict(long_text)

 assert result is not None
# Should not crash ```
### 15.2 tests/test_gap_detector.py
```python from src.gap_detector import GapDetector
def test_detects_missing_seed():

 detector = GapDetector("data/checklist/neurips_checklist.json")

 text = "We trained on CIFAR-10 using Adam optimizer for 50 epochs."

 gaps = detector.detect(text)

 seed_gap = next((g for g in gaps if "seed" in g["item"].lower()), None)

 assert seed_gap is not None

 assert seed_gap["status"] == "missing"
def test_detects_present_hyperparams():

 detector = GapDetector("data/checklist/neurips_checklist.json")

 text = "We used Adam with lr=0.001, beta1=0.9, beta2=0.999, weight_decay=1e-4."

 gaps = detector.detect(text)

 hp_gap = next((g for g in gaps if "hyperparameter" in g["item"].lower()), None)

 if hp_gap:



 assert hp_gap["status"] == "present" ```
### 15.3 tests/test_pipeline.py (Integration)
```python from fastapi.testclient import TestClient from api.main import app
client = TestClient(app)
def test_health():

 resp = client.get("/health")

 assert resp.status_code == 200

 assert resp.json()["status"] == "ok"
def test_analyze_text_endpoint():

 resp = client.post(



 "/api/v1/analyze/text",



 json={





 "text": "We trained ResNet-50 on ImageNet using SGD with lr=0.1, "









 "momentum=0.9, weight decay=1e-4. Random seed was 42. "









 "We used an 80/20 train/val split.",





 "generate_hints": False



 }

 )

 assert resp.status_code == 200

 data = resp.json()

 assert "reproducibility_score" in data

 assert "gaps" in data

 assert 0 <= data["score_out_of_100"] <= 100
def test_text_too_short():

 resp = client.post(



 "/api/v1/analyze/text",



 json={"text": "Too short.", "generate_hints": False}

 )

 assert resp.status_code == 422
# Validation error ```
Run tests: `pytest tests/ -v`
---
## 16. Deployment to HuggingFace Spaces
### 16.1 Dockerfile
```dockerfile FROM python:3.10-slim
WORKDIR /app
# Install system dependencies RUN apt-get update && apt-get install -y \

 gcc \

 g++ \

 && rm -rf /var/lib/apt/lists/*
# Copy requirements and install COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt
# Download NLTK data RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
# Copy all source code COPY . .
# Expose port EXPOSE 7860
# HuggingFace Spaces uses port 7860 CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"] ```
### 16.2 Streamlit Spaces (Alternative — Easier)
For a simpler deploy, use Streamlit SDK on HF Spaces instead of Docker: 1. Create new Space → SDK: Streamlit 2. Push code with `app/streamlit_app.py` as `app.py` at root 3. Add `requirements.txt` at root 4. Set `HF_TOKEN` in Space Secrets (Settings → Repository secrets) 5. The model checkpoint must be committed to the repo or loaded from HF Hub
### 16.3 Pushing Model to HF Hub
```python # Run this once after training: from huggingface_hub import HfApi api = HfApi() api.upload_folder(

 folder_path="models/scibert_finetuned",

 repo_id="xt67/reproducibility-predictor",
# Your HF username

 repo_type="model" ) ```
Then load in deployment: ```python model = AutoModelForSequenceClassification.from_pretrained("xt67/reproducibility-predictor") ```
---
## 17. README Template
```markdown # ML Paper Reproducibility Predictor
> Automatically detect reporting gaps in ML papers before submission.
**Live demo:** [huggingface.co/spaces/xt67/reproducibility-predictor](...)
 **Model:** [huggingface.co/xt67/reproducibility-predictor](...)
## What it does Paste the methods section of any ML paper → get: - A 0–100 reproducibility score (SciBERT-based) - A gap report (which of 43 NeurIPS checklist items are missing) - Sentence-level evidence highlights (SHAP attribution) - One-line fix suggestions per gap (Mistral-7B)
## Results | Model | AUROC | F1 (macro) | |-------|-------|------------| | TF-IDF + LR (baseline) | 0.68 | 0.63 | | SciBERT fine-tuned (ours) | 0.81 | 0.77 |
Gap detector precision on 30-paper manual eval: 0.72
## Architecture [Screenshot of architecture diagram from this conversation]
## Setup \`\`\`bash git clone https://github.com/xt67/reproducibility-predictor pip install -r requirements.txt cp .env.example .env
# Add your HF_TOKEN python scripts/download_data.py python scripts/train_model.py uvicorn api.main:app --reload \`\`\`
## Built by Rayan Rahman — Final Year B.Tech, Data Science
 [GitHub](https://github.com/xt67) · [LinkedIn](...) ```
---
## 18. Week-by-Week Checklist
``` Week 1
[ ] Repo created on GitHub



 [ ] Folder structure created



 [ ] requirements.txt committed



 [ ] PwC dataset downloaded



 [ ] data_pipeline.py complete



 [ ] 900/150/150 Parquet splits saved



 [ ] NeurIPS checklist JSON created (all 43 items)
Week 2
[ ] TF-IDF baseline: AUROC logged



 [ ] SciBERT fine-tuning complete



 [ ] AUROC > 0.78 on val set



 [ ] W&B training curves screenshot saved



 [ ] Model checkpoint saved to models/scibert_finetuned/
Week 3
[ ] gap_detector.py complete



 [ ] 30-paper manual eval done



 [ ] Precision > 0.68 confirmed



 [ ] Gap detector documented in README
Week 4
[ ] explainer.py complete (ablation-based SHAP)



 [ ] hint_generator.py complete



 [ ] pipeline.py complete (all modules integrated)



 [ ] End-to-end test: paste text → get JSON scorecard ✓
Week 5
[ ] FastAPI app running locally



 [ ] All 3 endpoints tested with curl



 [ ] Streamlit app running locally



 [ ] Full demo flow works end-to-end in browser
Week 6
[ ] Dockerfile tested locally



 [ ] Deployed to HuggingFace Spaces



 [ ] 3 example papers added to Streamlit "Try examples"



 [ ] README complete with results table + demo link



 [ ] Loom walkthrough recorded (2 min)



 [ ] Resume bullet written



 [ ] LinkedIn post drafted ```
---
## 19. Common Issues and Fixes
| Problem | Cause | Fix | |---------|-------|-----| | CUDA OOM during training | Batch size too large | Reduce BATCH_SIZE from 8 to 4 | | Methods section not found | Section header regex too strict | Add more patterns, fall back to first 2000 chars | | SHAP takes >60s | Too many sentences | Limit to first 25 sentences only | | HF Inference API returns 503 | Model loading (cold start) | Retry after 20s, up to 3 times | | Low AUROC (<0.72) | Wrong text being extracted | Print 5 samples and manually inspect | | Checklist similarity too high (everything "present") | Threshold too low | Raise SIMILARITY_THRESHOLD from 0.35 to 0.42 | | Streamlit slow on Spaces | Large model loading per session | Use `@st.cache_resource` on pipeline init |
---
*PRD Version 1.0 — Built for Claude Code. Every section above is a direct implementation instruction. Start at Section 3 (folder structure) and work top to bottom.*

