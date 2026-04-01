"""
Tests for HuggingFace Spaces Deployment (US-011).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== US-011: HuggingFace Spaces Deployment Tests ===")
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


# Test 1: Dockerfile exists
def test_dockerfile_exists():
    assert os.path.exists("Dockerfile"), "Dockerfile not found"

run_test("1. Dockerfile exists", test_dockerfile_exists)


# Test 2: Dockerfile has correct base image
def test_dockerfile_base_image():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "FROM python:" in content, "Missing Python base image"
    assert "3.11" in content or "3.10" in content, "Should use Python 3.10+"

run_test("2. Dockerfile has Python 3.10+ base", test_dockerfile_base_image)


# Test 3: Dockerfile exposes correct port
def test_dockerfile_port():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "EXPOSE 7860" in content, "Should expose port 7860 for HF Spaces"
    assert "7860" in content, "Port 7860 should be configured"

run_test("3. Dockerfile exposes port 7860", test_dockerfile_port)


# Test 4: Dockerfile runs Streamlit
def test_dockerfile_streamlit():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "streamlit" in content.lower(), "Should run streamlit"
    assert "app.py" in content, "Should run app.py"

run_test("4. Dockerfile runs Streamlit app", test_dockerfile_streamlit)


# Test 5: Dockerfile installs requirements
def test_dockerfile_requirements():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "requirements.txt" in content, "Should copy requirements.txt"
    assert "pip install" in content, "Should install dependencies"

run_test("5. Dockerfile installs requirements", test_dockerfile_requirements)


# Test 6: .dockerignore exists
def test_dockerignore_exists():
    assert os.path.exists(".dockerignore"), ".dockerignore not found"

run_test("6. .dockerignore exists", test_dockerignore_exists)


# Test 7: .dockerignore excludes large files
def test_dockerignore_content():
    with open(".dockerignore", "r") as f:
        content = f.read()
    assert "models/" in content or "scibert" in content.lower(), "Should exclude model checkpoints"
    assert "__pycache__" in content, "Should exclude Python cache"
    assert ".git" in content or "venv" in content, "Should exclude dev files"

run_test("7. .dockerignore excludes large files", test_dockerignore_content)


# Test 8: README has HuggingFace Spaces link
def test_readme_hf_link():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "huggingface.co/spaces" in content.lower(), "Should have HF Spaces link"

run_test("8. README has HuggingFace Spaces link", test_readme_hf_link)


# Test 9: README has demo screenshot reference
def test_readme_screenshot():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "screenshot" in content.lower() or "demo" in content.lower(), "Should reference demo screenshot"

run_test("9. README has demo screenshot reference", test_readme_screenshot)


# Test 10: README has Docker instructions
def test_readme_docker():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "docker" in content.lower(), "Should have Docker instructions"
    assert "docker build" in content.lower() or "docker run" in content.lower(), "Should have Docker commands"

run_test("10. README has Docker instructions", test_readme_docker)


# Test 11: requirements.txt is complete
def test_requirements_complete():
    with open("requirements.txt", "r") as f:
        content = f.read().lower()
    
    required = ["torch", "transformers", "streamlit", "fastapi", "sentence-transformers", "nltk"]
    for req in required:
        assert req in content, f"Missing {req} in requirements.txt"

run_test("11. requirements.txt has all dependencies", test_requirements_complete)


# Test 12: App runs on port 7860
def test_app_port_config():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "--server.port=7860" in content or "PORT" in content, "Should configure port 7860"

run_test("12. App configured for HF Spaces port", test_app_port_config)


# Test 13: Headless mode enabled
def test_headless_mode():
    with open("Dockerfile", "r") as f:
        content = f.read()
    assert "headless" in content.lower(), "Should run in headless mode"

run_test("13. Streamlit headless mode enabled", test_headless_mode)


# Summary
print()
print("=" * 50)
total = passed + failed
print(f"RESULT: {passed}/{total} tests passed, {failed} failed")
print("=" * 50)

# Deployment instructions
if failed == 0:
    print()
    print("✅ Ready for HuggingFace Spaces deployment!")
    print()
    print("To deploy:")
    print("1. Create a new Space at https://huggingface.co/new-space")
    print("2. Choose 'Docker' as the SDK")
    print("3. Push this repo to the Space")
    print("   git remote add hf https://huggingface.co/spaces/xt67/reproducibility-predictor")
    print("   git push hf main")

sys.exit(0 if failed == 0 else 1)
