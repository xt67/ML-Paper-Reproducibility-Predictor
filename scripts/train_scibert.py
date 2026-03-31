"""Train SciBERT classifier."""

import os
import pandas as pd

os.environ['TRANSFORMERS_CACHE'] = './models/.cache'

from src.classifier import ReproducibilityClassifier

# Load data
train_df = pd.read_parquet('data/processed/train.parquet')
val_df = pd.read_parquet('data/processed/val.parquet')

print(f'Training samples: {len(train_df)}')
print(f'Validation samples: {len(val_df)}')
label_col = 'label'
print(f'Label distribution: {train_df[label_col].value_counts().to_dict()}')
print()

# Initialize and train
classifier = ReproducibilityClassifier()
print('Starting training (this may take 30-60 min on CPU)...')
trainer = classifier.train(train_df, val_df, output_dir='models/scibert_finetuned')
print('Training complete!')

# Test prediction
test_text = "We used a learning rate of 0.001 and batch size 32. Random seed was set to 42."
result = classifier.predict(test_text)
print(f'\nTest prediction: {result}')

# Verify checkpoint
from pathlib import Path
checkpoint_dir = Path('models/scibert_finetuned')
print(f'\nCheckpoint files: {[f.name for f in checkpoint_dir.glob("*")]}')
