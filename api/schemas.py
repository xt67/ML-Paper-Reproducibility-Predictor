"""
Pydantic models for API request/response validation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class InputType(str, Enum):
    """Type of input for analysis."""
    pdf = "pdf"
    url = "url"
    arxiv = "arxiv"
    text = "text"


class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint with URL or arXiv ID."""
    url: Optional[HttpUrl] = Field(None, description="URL to PDF file")
    arxiv_id: Optional[str] = Field(None, description="arXiv paper ID (e.g., '2301.00001')")
    text: Optional[str] = Field(None, description="Direct methods section text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "2301.00001",
            }
        }


class GapItem(BaseModel):
    """A single gap detection result."""
    id: int
    item: str
    category: str
    severity: str
    status: str
    similarity_score: float
    best_matching_sentence: str
    hint: Optional[str] = None


class GapSummary(BaseModel):
    """Summary statistics for gap detection."""
    total_items: int
    present: int
    missing: int
    missing_high_severity: int
    missing_medium_severity: int
    missing_low_severity: int
    coverage_score: float
    weighted_score: float


class SentenceAttribution(BaseModel):
    """SHAP attribution for a single sentence."""
    rank: Optional[int] = None
    index: int
    sentence: str
    attribution: float
    normalized_score: float


class HighlightedSegment(BaseModel):
    """Text segment with color coding for UI."""
    text: str
    color: str  # "green", "yellow", "none"
    score: float


class ExplanationResult(BaseModel):
    """SHAP explanation results."""
    baseline_score: float
    sentences: list[SentenceAttribution]
    highlighted_text: list[HighlightedSegment]


class ClassificationResult(BaseModel):
    """Classification result from the model."""
    score: float = Field(..., ge=0, le=1, description="Reproducibility score (0-1)")
    label: int = Field(..., ge=0, le=1, description="Binary label (0 or 1)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in prediction")
    label_text: str = Field(..., description="Human-readable label")


class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint."""
    success: bool
    classification: ClassificationResult
    gaps: list[GapItem]
    gap_summary: GapSummary
    explanation: ExplanationResult
    methods_text: str = Field(..., description="Extracted methods section text")
    processing_time_seconds: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "classification": {
                    "score": 0.72,
                    "label": 1,
                    "confidence": 0.72,
                    "label_text": "Reproducible",
                },
                "gaps": [],
                "gap_summary": {
                    "total_items": 43,
                    "present": 30,
                    "missing": 13,
                    "missing_high_severity": 3,
                    "missing_medium_severity": 7,
                    "missing_low_severity": 3,
                    "coverage_score": 69.8,
                    "weighted_score": 72.5,
                },
                "explanation": {
                    "baseline_score": 0.72,
                    "sentences": [],
                    "highlighted_text": [],
                },
                "methods_text": "We trained our model using...",
                "processing_time_seconds": 15.3,
            }
        }


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    version: str
    models_loaded: dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
