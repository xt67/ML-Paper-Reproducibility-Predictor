"""
Health check router.
"""

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])

# API version
API_VERSION = "1.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and model status.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns API status and which models are loaded.
    """
    # Check model availability
    models_loaded = {
        "classifier": False,
        "gap_detector": False,
        "hint_generator": False,
    }
    
    try:
        from api.services import AnalysisService
        service = AnalysisService()
        models_loaded["classifier"] = service.classifier is not None
        models_loaded["gap_detector"] = service.gap_detector is not None
        models_loaded["hint_generator"] = service.hint_generator is not None
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        models_loaded=models_loaded,
    )
