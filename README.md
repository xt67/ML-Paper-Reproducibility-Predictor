# ML Paper Reproducibility Predictor

🔬 **[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/xt67/reproducibility-predictor)**

A web tool that analyzes the methods section of ML/AI research papers and outputs:
1. **Reproducibility Score** (0-100) — how likely is this paper to be reproducible?
2. **Gap Report** — which specific reporting items are missing (e.g., no random seed, no dataset split info)?
3. **Evidence Highlights** — which sentences drove the score, with SHAP-based attribution?
4. **Fix Hints** — one-line suggestions for each detected gap

## 🎯 Demo

![Reproducibility Predictor Demo](docs/demo-screenshot.png)

*Upload a PDF, enter an arXiv ID, or paste methods text to get instant reproducibility analysis.*

## 🚀 Quick Start

### Option 1: Use Online (Recommended)
Visit **[HuggingFace Spaces](https://huggingface.co/spaces/xt67/reproducibility-predictor)** — no setup required!

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/xt67/reproducibility-predictor.git
cd reproducibility-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the Streamlit app
streamlit run app.py
```

### Option 3: Docker
```bash
docker build -t reproducibility-predictor .
docker run -p 7860:7860 reproducibility-predictor
```

## 📁 Project Structure

```
reproducibility-predictor/
├── app.py                   # Streamlit frontend
├── api/                     # FastAPI backend
│   ├── main.py             # API entry point
│   ├── schemas.py          # Pydantic models
│   ├── services.py         # Analysis service
│   └── routers/            # API endpoints
├── src/                     # Core ML modules
│   ├── classifier.py       # SciBERT classifier
│   ├── gap_detector.py     # NeurIPS checklist checker
│   ├── explainer.py        # SHAP explanations
│   ├── hint_generator.py   # Fix suggestions
│   └── pdf_extractor.py    # PDF text extraction
├── data/                    # Datasets and checklist
├── models/                  # Trained model checkpoints
├── tests/                   # Unit tests
├── Dockerfile              # HuggingFace Spaces deployment
└── requirements.txt        # Python dependencies
```

## 🔌 API Usage

```python
import requests

# Analyze with arXiv ID
response = requests.post(
    "http://localhost:8000/analyze",
    json={"arxiv_id": "2301.00001"}
)
result = response.json()
print(f"Score: {result['classification']['score']}")
print(f"Missing items: {result['gap_summary']['missing']}")
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze paper (JSON: arxiv_id, url, or text) |
| `/analyze/upload` | POST | Analyze uploaded PDF |
| `/health` | GET | API health check |
| `/docs` | GET | Interactive API documentation |

## 🔧 Tech Stack

- **Backend:** FastAPI (Python 3.10+)
- **ML Model:** HuggingFace Transformers — allenai/scibert_scivocab_uncased
- **Explainability:** SHAP
- **Gap Detection:** sentence-transformers — all-MiniLM-L6-v2
- **Fix Hints:** HuggingFace Inference API — Mistral-7B-Instruct
- **PDF Parsing:** pdfminer.six + PyMuPDF
- **Frontend:** Streamlit
- **Deployment:** HuggingFace Spaces

## 📊 Model Performance

| Metric | Baseline (TF-IDF+LR) | SciBERT |
|--------|----------------------|---------|
| AUROC  | 0.68                 | **0.81** |
| F1     | 0.62                 | **0.76** |
| Precision | 0.64              | **0.78** |
| Recall | 0.61                 | **0.74** |

**Gap Detection:** 68% precision, 72% recall on manual evaluation (30 papers)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 ML Reproducibility Predictor                 │
├─────────────────────────────────────────────────────────────┤
│  Input: PDF / arXiv ID / URL / Text                         │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              PDF Extractor (PyMuPDF)                 │    │
│  │         → Methods section extraction                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐             │
│  │ SciBERT   │   │    Gap    │   │   SHAP    │             │
│  │Classifier │   │ Detector  │   │ Explainer │             │
│  │(0.81 AUC) │   │(MiniLM-L6)│   │(Ablation) │             │
│  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘             │
│        │               │               │                    │
│        ▼               ▼               ▼                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐            │
│  │  Score   │   │ Missing  │   │ Highlighted  │            │
│  │  0-100%  │   │  Items   │───│  Sentences   │            │
│  └──────────┘   └────┬─────┘   └──────────────┘            │
│                      │                                      │
│                      ▼                                      │
│               ┌────────────┐                                │
│               │    Hint    │                                │
│               │ Generator  │                                │
│               └────────────┘                                │
│                      │                                      │
│         ┌────────────┼────────────┐                        │
│         ▼            ▼            ▼                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                │
│  │ Streamlit │ │  FastAPI  │ │   JSON    │                │
│  │ Frontend  │ │  Backend  │ │  Export   │                │
│  └───────────┘ └───────────┘ └───────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Documentation

- **[API Documentation](docs/API.md)** — Endpoint reference, request/response schemas
- **[Technical Paper](docs/PAPER.md)** — Method details, results, architecture

## 📝 License

MIT License

## 👤 Author

Rayan Rahman
