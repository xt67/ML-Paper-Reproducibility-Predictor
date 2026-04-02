"""
FastAPI application for ML Paper Reproducibility Predictor.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analyze, health

# Create FastAPI app
app = FastAPI(
    title="ML Paper Reproducibility Predictor",
    description=(
        "Analyze ML paper methods sections for reproducibility.\n\n"
        "Features:\n"
        "- **Classification**: SciBERT-based reproducibility scoring (0-1)\n"
        "- **Gap Detection**: Identifies missing NeurIPS checklist items\n"
        "- **SHAP Explanation**: Highlights influential sentences\n"
        "- **Fix Hints**: Actionable suggestions for missing items"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analyze.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ML Paper Reproducibility Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
