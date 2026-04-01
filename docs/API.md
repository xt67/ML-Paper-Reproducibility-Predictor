# API Documentation

## Overview

The ML Paper Reproducibility Predictor API provides endpoints to analyze ML papers for reproducibility issues. The API accepts PDF uploads, arXiv IDs, URLs, or direct text input and returns comprehensive analysis including reproducibility scores, gap detection, SHAP explanations, and fix suggestions.

**Base URL:** `http://localhost:8000` (local) or HuggingFace Spaces URL

## Authentication

No authentication required for public endpoints.

---

## Endpoints

### Health Check

```
GET /health
```

Check API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "classifier": true,
    "gap_detector": true,
    "hint_generator": true
  }
}
```

---

### Analyze Paper (JSON)

```
POST /analyze
```

Analyze a paper using JSON input with arXiv ID, URL, or direct text.

**Request Body:**
```json
{
  "arxiv_id": "2301.00001",  // OR
  "url": "https://arxiv.org/pdf/2301.00001.pdf",  // OR
  "text": "We trained our model using Adam optimizer..."
}
```

**Response:**
```json
{
  "success": true,
  "classification": {
    "score": 0.72,
    "label": 1,
    "confidence": 0.72,
    "label_text": "Reproducible"
  },
  "gaps": [
    {
      "id": 11,
      "item": "The paper specifies random seeds used",
      "category": "experiments",
      "severity": "high",
      "status": "missing",
      "similarity_score": 0.21,
      "best_matching_sentence": "",
      "hint": "Specify random seeds for initialization and data shuffling."
    }
  ],
  "gap_summary": {
    "total_items": 43,
    "present": 30,
    "missing": 13,
    "missing_high_severity": 3,
    "missing_medium_severity": 7,
    "missing_low_severity": 3,
    "coverage_score": 69.8,
    "weighted_score": 72.5
  },
  "explanation": {
    "baseline_score": 0.72,
    "sentences": [
      {
        "rank": 1,
        "index": 2,
        "sentence": "We used random seed 42 for reproducibility.",
        "attribution": 0.15,
        "normalized_score": 0.85
      }
    ],
    "highlighted_text": [
      {
        "text": "We used random seed 42.",
        "color": "green",
        "score": 0.85
      }
    ]
  },
  "methods_text": "We trained our model...",
  "processing_time_seconds": 15.3
}
```

---

### Analyze Paper (PDF Upload)

```
POST /analyze/upload
Content-Type: multipart/form-data
```

Analyze a paper by uploading a PDF file.

**Request:**
- `file`: PDF file (multipart/form-data)

**Response:** Same as `/analyze`

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/analyze/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@paper.pdf"
```

---

### Analyze Text (Direct)

```
POST /analyze/text
```

Convenience endpoint for direct text analysis.

**Request Body:**
```json
{
  "text": "We trained our model using Adam optimizer with learning rate 0.001..."
}
```

**Response:** Same as `/analyze`

---

## Data Models

### ClassificationResult

| Field | Type | Description |
|-------|------|-------------|
| score | float | Reproducibility score (0-1) |
| label | int | Binary label (0=Not Reproducible, 1=Reproducible) |
| confidence | float | Model confidence (0-1) |
| label_text | string | Human-readable label |

### GapItem

| Field | Type | Description |
|-------|------|-------------|
| id | int | Checklist item ID |
| item | string | Checklist item description |
| category | string | Category (experiments, data, model, etc.) |
| severity | string | high, medium, or low |
| status | string | "present" or "missing" |
| similarity_score | float | Cosine similarity (0-1) |
| best_matching_sentence | string | Best matching sentence (if present) |
| hint | string | Fix suggestion (if missing) |

### GapSummary

| Field | Type | Description |
|-------|------|-------------|
| total_items | int | Total checklist items (43) |
| present | int | Items found in paper |
| missing | int | Items missing from paper |
| missing_high_severity | int | High severity missing items |
| missing_medium_severity | int | Medium severity missing items |
| missing_low_severity | int | Low severity missing items |
| coverage_score | float | Percentage of items present |
| weighted_score | float | Severity-weighted coverage |

### SentenceAttribution

| Field | Type | Description |
|-------|------|-------------|
| rank | int | Importance rank (1 = most influential) |
| index | int | Sentence index in original text |
| sentence | string | The sentence text |
| attribution | float | Raw SHAP attribution value |
| normalized_score | float | Normalized score (-1 to 1) |

### HighlightedSegment

| Field | Type | Description |
|-------|------|-------------|
| text | string | Sentence text |
| color | string | "green" (positive), "yellow" (negative), "none" (neutral) |
| score | float | Normalized attribution score |

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Must provide one of: arxiv_id, url, or text"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Analysis failed: [error message]"
}
```

---

## Rate Limits

- No rate limits on local deployment
- HuggingFace Spaces free tier: Subject to HF rate limits

---

## Example Code

### Python

```python
import requests

# Analyze arXiv paper
response = requests.post(
    "http://localhost:8000/analyze",
    json={"arxiv_id": "2301.00001"}
)
result = response.json()

print(f"Score: {result['classification']['score']:.1%}")
print(f"Label: {result['classification']['label_text']}")
print(f"Coverage: {result['gap_summary']['coverage_score']}%")
print(f"Missing high-severity: {result['gap_summary']['missing_high_severity']}")

# Get top influential sentences
for sent in result['explanation']['sentences'][:3]:
    print(f"  [{sent['rank']}] {sent['sentence'][:50]}...")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ arxiv_id: '2301.00001' })
});
const result = await response.json();
console.log(`Score: ${(result.classification.score * 100).toFixed(1)}%`);
```

### cURL

```bash
# Analyze with arXiv ID
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"arxiv_id": "2301.00001"}'

# Analyze with text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "We trained using Adam optimizer with lr=0.001..."}'

# Upload PDF
curl -X POST "http://localhost:8000/analyze/upload" \
  -F "file=@paper.pdf"
```

---

## Interactive Documentation

Access interactive API documentation at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
