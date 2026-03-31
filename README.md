# ML Paper Reproducibility Predictor

A web tool that analyzes the methods section of ML/AI research papers and outputs:
1. **Reproducibility Score** (0-100) — how likely is this paper to be reproducible?
2. **Gap Report** — which specific reporting items are missing (e.g., no random seed, no dataset split info)?
3. **Evidence Highlights** — which sentences drove the score, with SHAP-based attribution?
4. **Fix Hints** — one-line suggestions for each detected gap

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/reproducibility-predictor.git
cd reproducibility-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
reproducibility-predictor/
├── data/                    # Datasets and checklist
├── models/                  # Trained model checkpoints
├── src/                     # Core ML modules
├── api/                     # FastAPI backend
├── app/                     # Streamlit frontend
├── notebooks/               # Exploration & experiments
├── tests/                   # Unit tests
└── scripts/                 # Training & evaluation scripts
```

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
| AUROC  | 0.68                 | 0.81    |
| F1     | 0.62                 | 0.76    |

## 📝 License

MIT License

## 👤 Author

Rayan Rahman
