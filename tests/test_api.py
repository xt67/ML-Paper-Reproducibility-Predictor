"""
Tests for FastAPI backend endpoints.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== US-009: FastAPI Backend Tests ===")
print()

passed = 0
failed = 0
skipped = 0


def run_test(name: str, test_fn):
    """Run a single test and track results."""
    global passed, failed
    print(f"{name}...")
    try:
        test_fn()
        print(f"   ✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        traceback.print_exc()
        failed += 1



# Test 1: Import API modules
def test_api_imports():
    from api.main import app
    from api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
    from api.services import AnalysisService
    assert app is not None

run_test("1. API modules import", test_api_imports)


# Test 2: Test client creation
def test_client_creation():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    assert client is not None

run_test("2. Test client creation", test_client_creation)


# Test 3: Root endpoint
def test_root_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["docs"] == "/docs"

run_test("3. Root endpoint (/)", test_root_endpoint)


# Test 4: Health endpoint
def test_health_endpoint():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "models_loaded" in data

run_test("4. Health endpoint (/health)", test_health_endpoint)


# Test 5: OpenAPI schema
def test_openapi_schema():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/analyze" in schema["paths"]
    assert "/analyze/upload" in schema["paths"]
    assert "/health" in schema["paths"]

run_test("5. OpenAPI schema (/openapi.json)", test_openapi_schema)


# Test 6: Pydantic models validation
def test_pydantic_models():
    from api.schemas import (
        AnalyzeRequest,
        ClassificationResult,
        GapSummary,
    )
    
    # Test AnalyzeRequest
    req = AnalyzeRequest(arxiv_id="2301.00001")
    assert req.arxiv_id == "2301.00001"
    
    # Test with text
    req2 = AnalyzeRequest(text="Some methods text")
    assert req2.text == "Some methods text"
    
    # Test ClassificationResult
    cls_result = ClassificationResult(
        score=0.75,
        label=1,
        confidence=0.75,
        label_text="Reproducible"
    )
    assert cls_result.score == 0.75

run_test("6. Pydantic models validation", test_pydantic_models)


# Test 7: CORS enabled
def test_cors_enabled():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    
    # Preflight request
    response = client.options(
        "/analyze",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }
    )
    assert response.status_code == 200
    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers

run_test("7. CORS enabled", test_cors_enabled)


# Test 8: Analyze endpoint - missing input
def test_analyze_missing_input():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    
    # Empty request body should fail
    response = client.post("/analyze", json={})
    assert response.status_code == 400

run_test("8. Analyze endpoint - missing input", test_analyze_missing_input)


# Test 9: AnalysisService initialization
def test_analysis_service_init():
    from api.services import AnalysisService
    
    # Should initialize without errors
    service = AnalysisService()
    assert service.classifier is not None
    assert service.gap_detector is not None
    assert service.hint_generator is not None
    assert service.explainer is not None

run_test("9. AnalysisService initialization", test_analysis_service_init)


# Test 10: Analyze with text input
def test_analyze_with_text():
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    
    test_text = """
    We trained our model using the Adam optimizer with a learning rate of 0.001 
    and batch size of 32. The model was trained for 100 epochs on a single NVIDIA 
    V100 GPU. We used the CIFAR-10 dataset split into 45,000 training, 5,000 
    validation, and 10,000 test samples. Random seed was set to 42 for reproducibility.
    """
    
    response = client.post("/analyze", json={"text": test_text})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert data["success"] is True
    assert "classification" in data
    assert "gaps" in data
    assert "gap_summary" in data
    assert "explanation" in data
    assert "processing_time_seconds" in data
    
    # Validate classification
    assert 0 <= data["classification"]["score"] <= 1
    assert data["classification"]["label"] in [0, 1]
    
    # Validate gap summary
    assert data["gap_summary"]["total_items"] > 0
    assert "coverage_score" in data["gap_summary"]
    
    # Validate explanation
    assert "baseline_score" in data["explanation"]
    assert "sentences" in data["explanation"]
    assert "highlighted_text" in data["explanation"]

run_test("10. Analyze with text input", test_analyze_with_text)


# Test 11: Response time target (<30s)
def test_response_time():
    import time
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    
    test_text = "We used Adam optimizer with lr=0.001 and random seed 42."
    
    start = time.time()
    response = client.post("/analyze", json={"text": test_text})
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 30, f"Response took {elapsed:.1f}s, expected <30s"
    print(f"   Response time: {elapsed:.1f}s")

run_test("11. Response time target (<30s)", test_response_time)


# Summary
print()
print("=" * 50)
total = passed + failed
print(f"RESULT: {passed}/{total} tests passed, {failed} failed")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
