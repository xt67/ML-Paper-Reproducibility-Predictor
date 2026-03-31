"""Performance test for SHAP explainer."""

import time
from src.classifier import ReproducibilityClassifier
from src.explainer import SHAPExplainer

# Load model
classifier = ReproducibilityClassifier('models/scibert_finetuned')

# Generate test text with ~20 sentences
sentences = [
    'We trained our model using Adam optimizer with a learning rate of 0.001.',
    'The batch size was set to 32 for all experiments.',
    'We used a random seed of 42 to ensure reproducibility.',
    'Our experiments were conducted on a single NVIDIA V100 GPU.',
    'The training took approximately 24 hours to complete.',
    'We evaluated our model on three benchmark datasets.',
    'Our implementation is based on PyTorch 1.9.',
    'We used the standard train/val/test split.',
    'Hyperparameters were tuned using grid search.',
    'We report the mean and standard deviation over 5 runs.',
    'The embedding dimension was set to 768.',
    'We used dropout with probability 0.1.',
    'The model has approximately 110M parameters.',
    'Training was performed with mixed precision.',
    'We used gradient clipping with max norm 1.0.',
    'Early stopping was applied with patience 5.',
    'The learning rate was scheduled using cosine annealing.',
    'We used weight decay of 0.01 for regularization.',
    'Our code will be released upon publication.',
    'All experiments used the same random initialization.',
]
test_text = ' '.join(sentences)
print(f'Test text: {len(sentences)} sentences, {len(test_text)} chars')

# Clear cache to ensure fresh computation
explainer = SHAPExplainer(classifier, cache_dir='.cache/shap_perf_test')
explainer.clear_cache()

# Time the explanation
start = time.time()
result = explainer.explain(test_text, top_k=5)
elapsed = time.time() - start

print(f'Time for {len(sentences)} sentences: {elapsed:.2f}s')
print(f'Per sentence: {elapsed/len(sentences)*1000:.0f}ms')
print(f'Target: ~6s for 20 sentences')
status = 'PASS' if elapsed < 10 else 'FAIL'
print(f'Status: {status} (under 10s threshold)')
