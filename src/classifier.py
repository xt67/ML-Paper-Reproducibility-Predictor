"""
Reproducibility Classifier Module.
Contains both baseline (TF-IDF + Logistic Regression) and SciBERT classifiers.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Model configuration
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class BaselineClassifier:
    """
    TF-IDF + Logistic Regression baseline classifier.
    This establishes the performance floor that SciBERT must beat.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize baseline classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted = False
    
    def train(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """
        Train the baseline classifier.
        
        Args:
            train_df: Training data with 'methods_text' and 'label' columns
            val_df: Optional validation data for evaluation
        """
        # Fit TF-IDF and transform training data
        X_train = self.vectorizer.fit_transform(train_df["methods_text"])
        y_train = train_df["label"].values
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self._fitted = True
        
        # Evaluate on training data
        train_metrics = self._evaluate(X_train, y_train, "Train")
        
        # Evaluate on validation data if provided
        if val_df is not None:
            X_val = self.vectorizer.transform(val_df["methods_text"])
            y_val = val_df["label"].values
            val_metrics = self._evaluate(X_val, y_val, "Validation")
            return train_metrics, val_metrics
        
        return train_metrics
    
    def _evaluate(self, X, y_true, split_name: str) -> dict:
        """Evaluate and print metrics."""
        y_pred = self.classifier.predict(X)
        y_prob = self.classifier.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_prob),
            "f1": f1_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average="macro"),
        }
        
        print(f"\n{split_name} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  F1 (macro):{metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"\n{split_name} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Reproducible", "Reproducible"]))
        
        return metrics
    
    def predict(self, text: str) -> dict:
        """
        Predict reproducibility for a single text.
        
        Args:
            text: Methods section text
            
        Returns:
            Dictionary with score, label, confidence
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        label = int(proba[1] >= 0.5)
        
        return {
            "score": float(proba[1]),
            "label": label,
            "confidence": float(max(proba)),
            "probabilities": proba.tolist(),
        }
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch prediction for multiple texts."""
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X)
        
        results = []
        for proba in probas:
            label = int(proba[1] >= 0.5)
            results.append({
                "score": float(proba[1]),
                "label": label,
                "confidence": float(max(proba)),
            })
        
        return results
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate on test set."""
        X_test = self.vectorizer.transform(test_df["methods_text"])
        y_test = test_df["label"].values
        return self._evaluate(X_test, y_test, "Test")
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(path / "classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
    
    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(path / "classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)
        
        self._fitted = True


class ReproducibilityClassifier:
    """
    SciBERT-based reproducibility classifier.
    Fine-tuned on methods sections for binary classification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize SciBERT classifier.
        
        Args:
            model_path: Path to fine-tuned checkpoint. If None, loads pretrained.
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        
        if model_path:
            self.load(model_path)
    
    def _lazy_load(self):
        """Lazy load transformers to avoid import overhead."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        ).to(self.device)
        self._loaded = True
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str):
        """
        Fine-tune SciBERT on training data.
        
        Args:
            train_df: Training data
            val_df: Validation data
            output_dir: Directory to save checkpoints
            
        Returns:
            trainer: HuggingFace Trainer object for inspection
        """
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")
        
        # Load tokenizer and model with increased dropout for small dataset
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            hidden_dropout_prob=0.3,  # Increased for 22-sample dataset
            attention_probs_dropout_prob=0.3,
        ).to(self.device)
        
        # Compute class weights
        class_counts = train_df["label"].value_counts()
        total = len(train_df)
        class_weights = {
            0: total / (2 * class_counts.get(0, 1)),
            1: total / (2 * class_counts.get(1, 1)),
        }
        print(f"Class weights: {class_weights}")
        
        # Prepare datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples["methods_text"],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )
        
        train_dataset = Dataset.from_pandas(train_df[["methods_text", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["methods_text", "label"]])
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            
            return {
                "accuracy": accuracy_score(labels, predictions),
                "auroc": roc_auc_score(labels, probs),
                "f1": f1_score(labels, predictions, average="macro"),
            }
        
        # Check for W&B
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))
        if use_wandb:
            print("W&B logging enabled")
        
        # Training arguments with frequent eval for tiny dataset
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=4,  # Smaller for CPU
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,  # Effective batch=8
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE,
            eval_strategy="steps",  # Frequent eval for tiny dataset
            eval_steps=5,
            save_strategy="steps",
            save_steps=5,
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            greater_is_better=True,
            logging_dir=f"{output_dir}/logs",
            logging_steps=5,
            report_to="wandb" if use_wandb else "none",
            run_name="scibert-reproducibility" if use_wandb else None,
        )
        
        # Trainer with early stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train
        trainer.train()
        
        # Save best model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self._loaded = True
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def _sliding_window_predict(self, text: str) -> dict:
        """
        Predict using sliding window for long texts.
        Window: 512 tokens, Stride: 448 (64 overlap per CONTEXT.md)
        """
        import torch
        
        # Tokenize with overflow handling
        encoding = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            max_length=MAX_LENGTH,
            stride=MAX_LENGTH - 64,  # 512 - 64 = 448 stride
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        
        num_windows = encoding["input_ids"].shape[0]
        
        # Single window - use standard path
        if num_windows == 1:
            inputs = {
                "input_ids": encoding["input_ids"].to(self.device),
                "attention_mask": encoding["attention_mask"].to(self.device),
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()[0]
            
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            label = int(probs[1] >= 0.5)
            
            return {
                "score": float(probs[1]),
                "label": label,
                "confidence": float(max(probs)),
                "logits": logits.tolist(),
                "num_windows": 1,
            }
        
        # Multiple windows - aggregate via mean pooling
        all_logits = []
        for i in range(num_windows):
            inputs = {
                "input_ids": encoding["input_ids"][i:i+1].to(self.device),
                "attention_mask": encoding["attention_mask"][i:i+1].to(self.device),
            }
            with torch.no_grad():
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits.cpu())
        
        # Mean pool across windows
        stacked = torch.cat(all_logits, dim=0)
        mean_logits = stacked.mean(dim=0)
        
        probs = torch.softmax(mean_logits, dim=-1).numpy()
        label = int(probs[1] >= 0.5)
        
        return {
            "score": float(probs[1]),
            "label": label,
            "confidence": float(max(probs)),
            "logits": mean_logits.tolist(),
            "num_windows": num_windows,
        }
    
    def predict(self, text: str) -> dict:
        """
        Predict reproducibility for a single text.
        Uses sliding window for texts >512 tokens.
        
        Args:
            text: Methods section text
            
        Returns:
            Dictionary with score, label, confidence, logits, num_windows
        """
        self._lazy_load()
        return self._sliding_window_predict(text)
    
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batch prediction."""
        import torch
        from tqdm import tqdm
        
        self._lazy_load()
        
        results = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
            batch = texts[i:i + BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu()
            
            probs = torch.softmax(logits, dim=-1).numpy()
            
            for prob in probs:
                label = int(prob[1] >= 0.5)
                results.append({
                    "score": float(prob[1]),
                    "label": label,
                    "confidence": float(max(prob)),
                })
        
        return results
    
    def save(self, path: str):
        """Save model and tokenizer."""
        self._lazy_load()
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load model from checkpoint."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self._loaded = True


def train_baseline(
    train_path: str = "data/processed/train.parquet",
    val_path: str = "data/processed/val.parquet",
    test_path: str = "data/processed/test.parquet",
    save_path: str = "models/baseline",
):
    """
    Train and evaluate baseline classifier.
    """
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print("\nTraining baseline classifier...")
    classifier = BaselineClassifier()
    classifier.train(train_df, val_df)
    
    print("\nEvaluating on test set...")
    test_metrics = classifier.evaluate(test_df)
    
    print("\nSaving model...")
    classifier.save(save_path)
    
    return classifier, test_metrics


if __name__ == "__main__":
    train_baseline()
