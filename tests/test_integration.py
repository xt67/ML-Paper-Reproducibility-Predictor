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

# Test 7: SciBERT Classifier
print('7. SciBERT Classifier...')
try:
    from src.classifier import ReproducibilityClassifier
    clf = ReproducibilityClassifier('models/scibert_finetuned')
    result = clf.predict('We used learning rate 0.001 and batch size 32.')
    score = result["score"]
    windows = result.get("num_windows", 1)
    print(f'   ✓ Loaded & predicts: score={score:.3f}, windows={windows}')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 8: SciBERT Sliding Window
print('8. SciBERT Sliding Window...')
try:
    from src.classifier import ReproducibilityClassifier
    clf = ReproducibilityClassifier('models/scibert_finetuned')
    # Create long text that exceeds 512 tokens
    long_text = ' '.join(['This is a test sentence about machine learning experiments.'] * 200)
    result = clf.predict(long_text)
    windows = result.get("num_windows", 1)
    if windows > 1:
        print(f'   ✓ Sliding window works: {windows} windows used')
        passed += 1
    else:
        print(f'   ✗ FAILED: Expected multiple windows, got {windows}')
        failed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 9: SHAP Explainer
print('9. SHAP Explainer...')
try:
    from src.classifier import ReproducibilityClassifier
    from src.explainer import SHAPExplainer
    clf = ReproducibilityClassifier('models/scibert_finetuned')
    explainer = SHAPExplainer(clf, cache_dir='.cache/shap_test')
    result = explainer.explain('We used Adam. We set seed to 42.', top_k=2)
    n_sentences = len(result["sentences"])
    has_baseline = "baseline_score" in result
    print(f'   ✓ Works: {n_sentences} top sentences, baseline={result["baseline_score"]:.3f}')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 10: SHAP Highlighted Output
print('10. SHAP Highlighted Output...')
try:
    from src.classifier import ReproducibilityClassifier
    from src.explainer import SHAPExplainer
    clf = ReproducibilityClassifier('models/scibert_finetuned')
    explainer = SHAPExplainer(clf, cache_dir='.cache/shap_test')
    result = explainer.explain('Good sentence. Bad sentence. Neutral sentence.', top_k=3)
    highlighted = result["highlighted_text"]
    colors = [h["color"] for h in highlighted]
    valid_colors = all(c in ["green", "yellow", "none"] for c in colors)
    if valid_colors and len(highlighted) > 0:
        print(f'   ✓ Color coding works: {colors}')
        passed += 1
    else:
        print(f'   ✗ FAILED: Invalid colors {colors}')
        failed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

# Test 11: Full Pipeline Integration
print('11. Full Pipeline Integration...')
try:
    from src.classifier import ReproducibilityClassifier
    from src.gap_detector import GapDetector
    from src.hint_generator import HintGenerator
    from src.explainer import SHAPExplainer
    
    text = "We used Adam optimizer with lr=0.001 and random seed 42."
    
    # Classify
    clf = ReproducibilityClassifier('models/scibert_finetuned')
    classification = clf.predict(text)
    
    # Detect gaps
    detector = GapDetector()
    gaps = detector.detect(text)
    gap_summary = detector.summary(gaps)
    
    # Generate hints for a gap
    gen = HintGenerator()
    missing_gaps = [g for g in gaps if g["status"] == "MISSING"]
    if missing_gaps:
        hint = gen.generate_hint(missing_gaps[0])
    else:
        hint = "No gaps"
    
    # Explain
    explainer = SHAPExplainer(clf, cache_dir='.cache/shap_test')
    explanation = explainer.explain(text, top_k=3)
    
    print(f'   ✓ All 4 modules integrate: score={classification["score"]:.3f}, '
          f'coverage={gap_summary["coverage_score"]}%, explanations={len(explanation["sentences"])}')
    passed += 1
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    failed += 1

print()
print('=' * 50)
total = passed + failed
print(f'RESULT: {passed}/{total} tests passed, {failed} failed')
print('=' * 50)

sys.exit(0 if failed == 0 else 1)
