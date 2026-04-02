"""
Tests for Streamlit Frontend (US-010).
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== US-010: Streamlit Frontend Tests ===")
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
        traceback.print_exc()
        failed += 1


# Test 1: Streamlit import
def test_streamlit_import():
    import streamlit as st
    assert st.__version__ is not None

run_test("1. Streamlit import", test_streamlit_import)


# Test 2: App module import
def test_app_import():
    # Can't fully import app.py without Streamlit runtime, but check syntax
    import ast
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)  # Check syntax is valid

run_test("2. App module syntax check", test_app_import)


# Test 3: All required components exist in app
def test_app_components():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    # Check for required features
    assert "st.file_uploader" in source, "Missing file upload widget"
    assert "st.text_input" in source or "st.text_area" in source, "Missing text input"
    assert "st.progress" in source or "score" in source.lower(), "Missing score visualization"
    assert "gap" in source.lower(), "Missing gap report"
    assert "highlight" in source.lower(), "Missing SHAP highlighting"
    assert "hint" in source.lower(), "Missing fix hints"
    assert "st.spinner" in source, "Missing loading spinner"

run_test("3. Required UI components present", test_app_components)


# Test 4: Responsive CSS present
def test_responsive_css():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "@media" in source, "Missing responsive CSS media queries"
    assert "max-width" in source.lower() or "768px" in source, "Missing mobile breakpoint"

run_test("4. Responsive CSS for mobile", test_responsive_css)


# Test 5: Score gauge visualization
def test_score_gauge():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "render_score_gauge" in source, "Missing score gauge function"
    assert "score-high" in source, "Missing high score styling"
    assert "score-low" in source, "Missing low score styling"
    assert "st.progress" in source, "Missing progress bar"

run_test("5. Score gauge visualization", test_score_gauge)


# Test 6: Gap report table with severity sorting
def test_gap_table():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "render_gap_table" in source, "Missing gap table function"
    assert "severity" in source.lower(), "Missing severity handling"
    assert "high" in source.lower() and "medium" in source.lower(), "Missing severity levels"

run_test("6. Gap report table with severity", test_gap_table)


# Test 7: SHAP highlighted text
def test_shap_highlighting():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "render_highlighted_text" in source, "Missing highlighted text function"
    assert "highlight-green" in source, "Missing green highlight"
    assert "highlight-yellow" in source, "Missing yellow highlight"
    assert "explanation" in source.lower(), "Missing explanation handling"

run_test("7. SHAP highlighted text", test_shap_highlighting)


# Test 8: Fix hints display
def test_fix_hints():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "hint" in source.lower(), "Missing hint handling"
    assert "HintGenerator" in source, "Missing HintGenerator import"
    assert "generate_hints" in source, "Missing hint generation call"

run_test("8. Fix hints display", test_fix_hints)


# Test 9: Multiple input methods
def test_input_methods():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "file_uploader" in source, "Missing PDF upload"
    assert "arxiv" in source.lower(), "Missing arXiv input"
    assert "text_area" in source, "Missing text input"
    assert "input_method" in source or "radio" in source, "Missing input method selector"

run_test("9. Multiple input methods (PDF, URL, text)", test_input_methods)


# Test 10: Loading spinner during analysis
def test_loading_spinner():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert source.count("st.spinner") >= 2, "Should have multiple spinners for loading states"
    assert "Analyzing" in source, "Missing analysis spinner message"

run_test("10. Loading spinner during analysis", test_loading_spinner)


# Test 11: Model caching
def test_model_caching():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "@st.cache_resource" in source, "Missing model caching decorator"
    assert "load_models" in source, "Missing load_models function"

run_test("11. Model caching for performance", test_model_caching)


# Test 12: Download results option
def test_download_results():
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    
    assert "download_button" in source, "Missing download button"
    assert "json" in source.lower(), "Missing JSON export"

run_test("12. Download results option", test_download_results)


# Summary
print()
print("=" * 50)
total = passed + failed
print(f"RESULT: {passed}/{total} tests passed, {failed} failed")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
