"""
SHAP Explainability Module.
Sentence-level attribution using leave-one-out ablation.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class SHAPExplainer:
    """
    Ablation-based SHAP approximation for sentence-level attribution.
    
    Uses leave-one-out ablation: for each sentence, remove it and measure
    how much the prediction changes. Positive attribution means the sentence
    increases the reproducibility score.
    """
    
    def __init__(self, classifier, cache_dir: Optional[str] = ".cache/shap"):
        """
        Initialize SHAP explainer.
        
        Args:
            classifier: A classifier with predict(text) -> {"score": float, ...}
            cache_dir: Directory for caching results (None to disable)
        """
        self.classifier = classifier
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate hash key for caching."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[dict]:
        """Load cached result if exists."""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, result: dict):
        """Save result to cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except IOError:
            pass  # Silently fail on cache write errors
    
    def explain(self, text: str, top_k: int = 5) -> dict:
        """
        Compute sentence-level SHAP values via leave-one-out ablation.
        
        Args:
            text: Input text to explain
            top_k: Number of top sentences to return (default: 5)
            
        Returns:
            Dictionary with:
            - baseline_score: Original prediction score
            - sentences: Top-k most influential sentences with scores
            - all_attributions: All sentences with their attribution scores
            - highlighted_text: Text segments with color hints for UI
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) == 0:
            result = {
                "baseline_score": 0.5,
                "sentences": [],
                "all_attributions": [],
                "highlighted_text": [],
            }
            self._save_to_cache(cache_key, result)
            return result
        
        # Get baseline prediction
        baseline = self.classifier.predict(text)
        baseline_score = baseline["score"]
        
        # Compute ablation scores (leave-one-out)
        attributions = []
        for i, sent in enumerate(sentences):
            # Create text without this sentence
            ablated_sentences = sentences[:i] + sentences[i+1:]
            ablated_text = " ".join(ablated_sentences)
            
            if ablated_text.strip():
                ablated = self.classifier.predict(ablated_text)
                ablated_score = ablated["score"]
            else:
                # If removing this sentence leaves nothing, assume neutral
                ablated_score = 0.5
            
            # Attribution = how much removing this sentence changes score
            # Positive = sentence increases reproducibility score (good)
            # Negative = sentence decreases reproducibility score (bad)
            attribution = baseline_score - ablated_score
            
            attributions.append({
                "index": i,
                "sentence": sent,
                "attribution": attribution,
            })
        
        # Normalize to [-1, 1]
        max_abs = max((abs(a["attribution"]) for a in attributions), default=1.0)
        if max_abs == 0:
            max_abs = 1.0
        
        for a in attributions:
            a["normalized_score"] = a["attribution"] / max_abs
        
        # Sort by absolute attribution (most influential first)
        attributions_sorted = sorted(
            attributions, 
            key=lambda x: abs(x["attribution"]), 
            reverse=True
        )
        
        # Get top-k and add rank
        top_sentences = attributions_sorted[:top_k]
        for rank, sent in enumerate(top_sentences, 1):
            sent["rank"] = rank
        
        # Generate highlighted text (in original order)
        highlighted = self._generate_highlighted_text(sentences, attributions)
        
        result = {
            "baseline_score": baseline_score,
            "sentences": top_sentences,
            "all_attributions": attributions,  # Original order
            "highlighted_text": highlighted,
        }
        
        # Cache result
        self._save_to_cache(cache_key, result)
        
        return result
    
    def _generate_highlighted_text(self, sentences: list, attributions: list) -> list:
        """
        Generate text segments with color coding for UI.
        
        Colors:
        - green: Positive attribution (increases reproducibility score)
        - yellow: Negative attribution (decreases reproducibility score)  
        - none: Near-zero attribution (neutral)
        """
        highlighted = []
        
        # Threshold for "neutral" - within 10% of max
        max_abs = max((abs(a["normalized_score"]) for a in attributions), default=0)
        neutral_threshold = 0.1 * max_abs if max_abs > 0 else 0.1
        
        for attr in attributions:
            score = attr["normalized_score"]
            
            if abs(score) < neutral_threshold:
                color = "none"
            elif score > 0:
                color = "green"
            else:
                color = "yellow"
            
            highlighted.append({
                "text": attr["sentence"],
                "color": color,
                "score": score,
            })
        
        return highlighted
    
    def clear_cache(self):
        """Clear all cached results."""
        if self.cache_dir and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()


def analyze_with_explanation(
    text: str,
    classifier,
    cache_dir: str = ".cache/shap",
    top_k: int = 5,
) -> dict:
    """
    Convenience function to analyze text with SHAP explanation.
    
    Args:
        text: Methods section text to analyze
        classifier: Classifier with predict() method
        cache_dir: Cache directory for SHAP results
        top_k: Number of top influential sentences to return
        
    Returns:
        Combined analysis with classification and explanation
    """
    # Get classification
    classification = classifier.predict(text)
    
    # Get SHAP explanation
    explainer = SHAPExplainer(classifier, cache_dir=cache_dir)
    explanation = explainer.explain(text, top_k=top_k)
    
    return {
        "classification": classification,
        "explanation": explanation,
    }


if __name__ == "__main__":
    # Demo/test
    import sys
    sys.path.insert(0, ".")
    
    from src.classifier import ReproducibilityClassifier
    
    # Load trained model
    try:
        classifier = ReproducibilityClassifier("models/scibert_finetuned")
        print("Loaded trained SciBERT model")
    except Exception:
        print("No trained model found, using untrained model")
        classifier = ReproducibilityClassifier()
    
    # Test text
    test_text = """
    We trained our model using Adam optimizer with a learning rate of 0.001.
    The batch size was set to 32 for all experiments.
    We used a random seed of 42 to ensure reproducibility.
    Our experiments were conducted on a single NVIDIA V100 GPU.
    The training took approximately 24 hours to complete.
    We did not release our code or data.
    """
    
    print("\n" + "="*60)
    print("SHAP Explainer Demo")
    print("="*60)
    
    explainer = SHAPExplainer(classifier)
    result = explainer.explain(test_text.strip(), top_k=5)
    
    print(f"\nBaseline score: {result['baseline_score']:.3f}")
    print(f"\nTop {len(result['sentences'])} influential sentences:")
    
    for sent in result["sentences"]:
        sign = "+" if sent["normalized_score"] > 0 else ""
        print(f"  [{sent['rank']}] ({sign}{sent['normalized_score']:.2f}) {sent['sentence'][:60]}...")
    
    print(f"\nHighlighted text (color coding):")
    for item in result["highlighted_text"]:
        color_emoji = {"green": "🟢", "yellow": "🟡", "none": "⚪"}[item["color"]]
        print(f"  {color_emoji} {item['text'][:50]}...")
