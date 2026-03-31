"""Integration test for milestone audit."""
import sys

print('=== MILESTONE AUDIT: Integration Test ===')
print()

passed = 0
failed = 0

# Test 1: Data Pipeline
print('1. Data Pipeline...')
try:
    import pandas as pd
    train = pd.read_parquet('data/processed/train.parquet')
    val = pd.read_parquet('data/processed/val.parquet')
    test = pd.read_parquet('data/processed/test.parquet')
    print(f'   ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)} samples')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 2: PDF Extractor
print('2. PDF Extractor...')
try:
    from src.pdf_extractor import extract_from_pdf, find_methods_section
    print('   ✓ Module loads, functions available')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 3: Baseline Classifier
print('3. Baseline Classifier...')
try:
    from src.classifier import BaselineClassifier
    clf = BaselineClassifier()
    clf.load('models/baseline')
    result = clf.predict('We used learning rate 0.001 and batch size 32.')
    score = result["score"]
    print(f'   ✓ Loaded & predicts: score={score:.3f}')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 4: Checklist JSON
print('4. NeurIPS Checklist...')
try:
    import json
    with open('data/checklist/neurips_checklist.json') as f:
        checklist = json.load(f)
    print(f'   ✓ {len(checklist)} items loaded')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 5: Gap Detector
print('5. Gap Detector...')
try:
    from src.gap_detector import GapDetector
    detector = GapDetector()
    gaps = detector.detect('We used Adam optimizer with lr=0.001.')
    summary = detector.summary(gaps)
    cov = summary["coverage_score"]
    miss = summary["missing"]
    print(f'   ✓ Coverage: {cov}%, {miss} gaps detected')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 6: Hint Generator
print('6. Hint Generator...')
try:
    from src.hint_generator import HintGenerator
    gen = HintGenerator()
    hint = gen.generate_hint({'item': 'Random seed', 'severity': 'high', 'category': 'experiments'})
    print(f'   ✓ Generates hints: "{hint[:40]}..."')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

print()
print('=' * 50)
print(f'RESULT: {passed}/6 tests passed, {failed} failed')
print('=' * 50)

sys.exit(0 if failed == 0 else 1)
