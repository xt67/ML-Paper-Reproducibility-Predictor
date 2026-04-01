"""
Analysis router for reproducibility prediction.
"""

import time
from typing import Optional

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassificationResult,
    ErrorResponse,
    ExplanationResult,
    GapItem,
    GapSummary,
    HighlightedSegment,
    SentenceAttribution,
)
from api.services import AnalysisService

router = APIRouter(tags=["analysis"])

# Lazy-loaded service
_service: Optional[AnalysisService] = None


def get_service() -> AnalysisService:
    """Get or create the analysis service singleton."""
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service


def _build_response(result: dict, methods_text: str, processing_time: float) -> AnalyzeResponse:
    """Build AnalyzeResponse from analysis result."""
    classification = ClassificationResult(
        score=result["classification"]["score"],
        label=result["classification"]["label"],
        confidence=result["classification"]["confidence"],
        label_text="Reproducible" if result["classification"]["label"] == 1 else "Not Reproducible",
    )
    
    gaps = [
        GapItem(
            id=g["id"],
            item=g["item"],
            category=g["category"],
            severity=g["severity"],
            status=g["status"],
            similarity_score=g["similarity_score"],
            best_matching_sentence=g["best_matching_sentence"],
            hint=g.get("hint"),
        )
        for g in result["gaps"]
    ]
    
    gap_summary = GapSummary(**result["gap_summary"])
    
    sentences = [
        SentenceAttribution(
            rank=s.get("rank"),
            index=s["index"],
            sentence=s["sentence"],
            attribution=s["attribution"],
            normalized_score=s["normalized_score"],
        )
        for s in result["explanation"]["sentences"]
    ]
    
    highlighted = [
        HighlightedSegment(
            text=h["text"],
            color=h["color"],
            score=h["score"],
        )
        for h in result["explanation"]["highlighted_text"]
    ]
    
    explanation = ExplanationResult(
        baseline_score=result["explanation"]["baseline_score"],
        sentences=sentences,
        highlighted_text=highlighted,
    )
    
    return AnalyzeResponse(
        success=True,
        classification=classification,
        gaps=gaps,
        gap_summary=gap_summary,
        explanation=explanation,
        methods_text=methods_text[:2000],
        processing_time_seconds=round(processing_time, 2),
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Analyze paper reproducibility",
    description="Analyze an ML paper's methods section for reproducibility. "
    "Accepts JSON body with url, arxiv_id, or text field.",
)
async def analyze_paper(
    request: AnalyzeRequest = Body(...),
) -> AnalyzeResponse:
    """
    Analyze paper reproducibility with JSON input.
    
    Accepts JSON body with one of:
    - url: URL to PDF
    - arxiv_id: arXiv paper ID  
    - text: Direct methods section text
    
    Returns reproducibility score, gap report, SHAP explanations, and fix hints.
    """
    start_time = time.time()
    service = get_service()
    
    try:
        methods_text = None
        
        if request.arxiv_id:
            methods_text = service.extract_from_arxiv(request.arxiv_id)
        elif request.url:
            methods_text = service.extract_from_url(str(request.url))
        elif request.text:
            methods_text = request.text
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide one of: arxiv_id, url, or text",
            )
        
        if not methods_text or len(methods_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from input (minimum 50 characters)",
            )
        
        result = service.analyze(methods_text)
        processing_time = time.time() - start_time
        
        return _build_response(result, methods_text, processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/analyze/upload",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Analyze paper from PDF upload",
    description="Upload a PDF file for reproducibility analysis.",
)
async def analyze_upload(
    file: UploadFile = File(..., description="PDF file to analyze"),
) -> AnalyzeResponse:
    """
    Analyze paper reproducibility from PDF upload.
    """
    start_time = time.time()
    service = get_service()
    
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be a PDF",
            )
        
        methods_text = await service.extract_from_upload(file)
        
        if not methods_text or len(methods_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF (minimum 50 characters)",
            )
        
        result = service.analyze(methods_text)
        processing_time = time.time() - start_time
        
        return _build_response(result, methods_text, processing_time)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/analyze/text",
    response_model=AnalyzeResponse,
    summary="Analyze text directly",
    description="Analyze methods section text directly without PDF extraction.",
)
async def analyze_text(
    text: str = Body(..., embed=True),
) -> AnalyzeResponse:
    """Convenience endpoint for direct text analysis."""
    request = AnalyzeRequest(text=text)
    return await analyze_paper(request=request)

