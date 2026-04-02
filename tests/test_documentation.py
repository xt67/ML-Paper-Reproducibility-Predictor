"""
Tests for Documentation (US-012).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== US-012: Documentation Tests ===")
print()

passed = 0
failed = 0


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
        failed += 1


# Test 1: README exists and is comprehensive
def test_readme_comprehensive():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check required sections
    assert "# ML Paper Reproducibility Predictor" in content, "Missing title"
    assert "Quick Start" in content, "Missing Quick Start section"
    assert "Installation" in content.lower() or "install" in content.lower(), "Missing installation"
    assert "docker" in content.lower(), "Missing Docker instructions"

run_test("1. README is comprehensive", test_readme_comprehensive)


# Test 2: README has architecture diagram
def test_readme_architecture():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "Architecture" in content or "architecture" in content, "Missing architecture section"
    assert "┌" in content or "```" in content, "Missing diagram"
    assert "SciBERT" in content, "Missing SciBERT in architecture"

run_test("2. README has architecture diagram", test_readme_architecture)


# Test 3: API documentation exists
def test_api_docs_exist():
    assert os.path.exists("docs/API.md"), "docs/API.md not found"
    
    with open("docs/API.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert len(content) > 1000, "API docs too short"

run_test("3. API documentation exists", test_api_docs_exist)


# Test 4: API docs cover all endpoints
def test_api_docs_endpoints():
    with open("docs/API.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    endpoints = ["/analyze", "/health", "/analyze/upload"]
    for endpoint in endpoints:
        assert endpoint in content, f"Missing endpoint: {endpoint}"

run_test("4. API docs cover all endpoints", test_api_docs_endpoints)


# Test 5: API docs have request/response examples
def test_api_docs_examples():
    with open("docs/API.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "```json" in content, "Missing JSON examples"
    assert "```python" in content or "```bash" in content, "Missing code examples"
    assert "curl" in content.lower(), "Missing curl examples"

run_test("5. API docs have request/response examples", test_api_docs_examples)


# Test 6: Paper draft exists
def test_paper_exists():
    assert os.path.exists("docs/PAPER.md"), "docs/PAPER.md not found"
    
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert len(content) > 5000, "Paper too short (should be ~4 pages)"

run_test("6. Paper draft exists", test_paper_exists)


# Test 7: Paper has required sections
def test_paper_sections():
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = ["Introduction", "Method", "Results", "Conclusion"]
    for section in sections:
        assert section in content, f"Missing section: {section}"

run_test("7. Paper has intro/method/results/conclusion", test_paper_sections)


# Test 8: Paper has baseline comparison table
def test_paper_baseline_comparison():
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "Baseline" in content, "Missing baseline comparison"
    assert "TF-IDF" in content, "Missing TF-IDF baseline"
    assert "SciBERT" in content, "Missing SciBERT results"
    assert "0.81" in content or "0.68" in content, "Missing AUROC values"

run_test("8. Paper has baseline comparison table", test_paper_baseline_comparison)


# Test 9: Paper has SHAP examples
def test_paper_shap_examples():
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "SHAP" in content, "Missing SHAP discussion"
    assert "ablation" in content.lower() or "attribution" in content.lower(), "Missing SHAP method"

run_test("9. Paper has SHAP examples", test_paper_shap_examples)


# Test 10: Paper has gap detection accuracy
def test_paper_gap_accuracy():
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "Gap Detection" in content or "gap detection" in content, "Missing gap detection section"
    assert "precision" in content.lower(), "Missing precision metric"
    assert "0.68" in content or "68%" in content, "Missing gap detection accuracy"

run_test("10. Paper has gap detection accuracy", test_paper_gap_accuracy)


# Test 11: Resume bullet exists
def test_resume_bullet():
    with open("docs/PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "Resume" in content or "resume" in content, "Missing resume bullet"
    assert "0.81" in content, "Missing quantified metric"
    assert "SciBERT" in content or "PyTorch" in content, "Missing tech stack"

run_test("11. Resume bullet with metrics and tech stack", test_resume_bullet)


# Test 12: README links to documentation
def test_readme_links_docs():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "docs/API.md" in content or "API Documentation" in content, "Missing API doc link"
    assert "docs/PAPER.md" in content or "Technical Paper" in content, "Missing paper link"

run_test("12. README links to documentation", test_readme_links_docs)


# Summary
print()
print("=" * 50)
total = passed + failed
print(f"RESULT: {passed}/{total} tests passed, {failed} failed")
print("=" * 50)

if failed == 0:
    print()
    print("✅ Documentation complete!")
    print()
    print("Deliverables:")
    print("  - README.md: Installation, usage, architecture")
    print("  - docs/API.md: Full endpoint documentation")
    print("  - docs/PAPER.md: 4-page technical paper draft")

sys.exit(0 if failed == 0 else 1)
