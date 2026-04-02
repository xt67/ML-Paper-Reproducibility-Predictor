"""
Gap Detector Module.
Identifies missing reproducibility checklist items using sentence-transformer similarity.
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


def simple_sent_tokenize(text: str) -> list[str]:
    """Simple sentence tokenizer as fallback."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def sent_tokenize(text: str) -> list[str]:
    """Tokenize text into sentences with NLTK fallback."""
    try:
        import nltk
        # Try downloading resources
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
        return nltk.sent_tokenize(text)
    except Exception:
        # Fallback to simple regex-based tokenizer
        return simple_sent_tokenize(text)


# Configuration
SIMILARITY_THRESHOLD = 0.35  # Below this = item is missing
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# Default checklist path
DEFAULT_CHECKLIST_PATH = Path(__file__).parent.parent / "data" / "checklist" / "neurips_checklist.json"


class GapDetector:
    """
    Detects missing reproducibility checklist items in methods sections.
    Uses sentence-transformer embeddings and cosine similarity.
    """
    
    def __init__(
        self, 
        checklist_path: Optional[str] = None,
        model_name: str = SENTENCE_MODEL,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Initialize gap detector.
        
        Args:
            checklist_path: Path to checklist JSON. Uses default if None.
            model_name: Sentence transformer model to use.
            threshold: Similarity threshold. Below = missing.
        """
        self.threshold = threshold
        
        # Load checklist
        checklist_path = checklist_path or str(DEFAULT_CHECKLIST_PATH)
        with open(checklist_path, "r", encoding="utf-8") as f:
            self.checklist = json.load(f)
        
        print(f"Loaded {len(self.checklist)} checklist items")
        
        # Load sentence transformer
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Pre-encode checklist items (do once at init)
        checklist_texts = [item["item"] for item in self.checklist]
        self.checklist_embeddings = self.model.encode(
            checklist_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        print("Checklist embeddings cached")
    
    def detect(self, methods_text: str) -> list[dict]:
        """
        Detect missing reproducibility items in the methods text.
        
        Args:
            methods_text: The methods section text to analyze
            
        Returns:
            List of dicts, one per checklist item:
            {
                "id": int,
                "item": str,
                "category": str,
                "severity": str,
                "status": "missing" | "present",
                "similarity_score": float,
                "best_matching_sentence": str
            }
        """
        # Split text into sentences
        sentences = sent_tokenize(methods_text)
        
        if not sentences:
            # Return all items as missing if no sentences
            return [
                {
                    "id": item["id"],
                    "item": item["item"],
                    "category": item.get("category", "unknown"),
                    "severity": item.get("severity", "medium"),
                    "status": "missing",
                    "similarity_score": 0.0,
                    "best_matching_sentence": "",
                }
                for item in self.checklist
            ]
        
        # Encode sentences
        sentence_embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        
        # Compute similarities between all checklist items and all sentences
        # Shape: (num_checklist_items, num_sentences)
        similarities = util.cos_sim(self.checklist_embeddings, sentence_embeddings)
        similarities = similarities.cpu().numpy()
        
        results = []
        
        for i, item in enumerate(self.checklist):
            # Get max similarity for this checklist item across all sentences
            max_sim_idx = np.argmax(similarities[i])
            max_sim = float(similarities[i, max_sim_idx])
            
            # Determine status
            if max_sim >= self.threshold:
                status = "present"
                best_sentence = sentences[max_sim_idx]
            else:
                status = "missing"
                best_sentence = ""
            
            results.append({
                "id": item["id"],
                "item": item["item"],
                "category": item.get("category", "unknown"),
                "severity": item.get("severity", "medium"),
                "status": status,
                "similarity_score": max_sim,
                "best_matching_sentence": best_sentence,
            })
        
        return results
    
    def summary(self, gaps: list[dict]) -> dict:
        """
        Compute summary statistics from gap detection results.
        
        Args:
            gaps: Output from detect()
            
        Returns:
            Summary dict with counts and coverage score
        """
        total = len(gaps)
        present = sum(1 for g in gaps if g["status"] == "present")
        missing = total - present
        
        # Count by severity
        missing_high = sum(1 for g in gaps if g["status"] == "missing" and g["severity"] == "high")
        missing_medium = sum(1 for g in gaps if g["status"] == "missing" and g["severity"] == "medium")
        missing_low = sum(1 for g in gaps if g["status"] == "missing" and g["severity"] == "low")
        
        # Coverage score (percentage of items present)
        coverage_score = (present / total * 100) if total > 0 else 0
        
        # Weighted score (high severity items count more)
        severity_weights = {"high": 3, "medium": 2, "low": 1}
        max_weighted = sum(severity_weights.get(g["severity"], 1) for g in gaps)
        present_weighted = sum(
            severity_weights.get(g["severity"], 1) 
            for g in gaps 
            if g["status"] == "present"
        )
        weighted_score = (present_weighted / max_weighted * 100) if max_weighted > 0 else 0
        
        return {
            "total_items": total,
            "present": present,
            "missing": missing,
            "missing_high_severity": missing_high,
            "missing_medium_severity": missing_medium,
            "missing_low_severity": missing_low,
            "coverage_score": round(coverage_score, 1),
            "weighted_score": round(weighted_score, 1),
        }
    
    def get_missing_items(self, gaps: list[dict], severity: Optional[str] = None) -> list[dict]:
        """
        Get only missing items, optionally filtered by severity.
        
        Args:
            gaps: Output from detect()
            severity: Filter to this severity ("high", "medium", "low") or None for all
            
        Returns:
            List of missing items
        """
        missing = [g for g in gaps if g["status"] == "missing"]
        
        if severity:
            missing = [g for g in missing if g["severity"] == severity]
        
        # Sort by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        missing.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["id"]))
        
        return missing
    
    def format_report(self, gaps: list[dict], methods_text: str = "") -> str:
        """
        Format gap detection results as a human-readable report.
        
        Args:
            gaps: Output from detect()
            methods_text: Original text (for context)
            
        Returns:
            Formatted report string
        """
        summary = self.summary(gaps)
        missing = self.get_missing_items(gaps)
        
        lines = [
            "=" * 60,
            "REPRODUCIBILITY GAP REPORT",
            "=" * 60,
            "",
            f"Coverage Score: {summary['coverage_score']:.1f}%",
            f"Weighted Score: {summary['weighted_score']:.1f}%",
            "",
            f"Items Present: {summary['present']}/{summary['total_items']}",
            f"Items Missing: {summary['missing']}",
            f"  - High Severity: {summary['missing_high_severity']}",
            f"  - Medium Severity: {summary['missing_medium_severity']}",
            f"  - Low Severity: {summary['missing_low_severity']}",
            "",
        ]
        
        if missing:
            lines.extend([
                "-" * 60,
                "MISSING ITEMS (by severity)",
                "-" * 60,
                "",
            ])
            
            current_severity = None
            for item in missing:
                if item["severity"] != current_severity:
                    current_severity = item["severity"]
                    lines.append(f"\n[{current_severity.upper()} SEVERITY]")
                
                lines.append(f"  • [{item['id']}] {item['item']}")
                lines.append(f"    Similarity: {item['similarity_score']:.3f}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def analyze_paper(methods_text: str, checklist_path: Optional[str] = None) -> tuple[list[dict], dict, str]:
    """
    Convenience function to analyze a paper's methods section.
    
    Args:
        methods_text: The methods section text
        checklist_path: Optional custom checklist path
        
    Returns:
        Tuple of (gaps, summary, report)
    """
    detector = GapDetector(checklist_path=checklist_path)
    gaps = detector.detect(methods_text)
    summary = detector.summary(gaps)
    report = detector.format_report(gaps, methods_text)
    
    return gaps, summary, report


if __name__ == "__main__":
    import sys
    
    # Test with sample text or file
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], "r") as f:
            text = f.read()
    else:
        # Sample methods text
        text = """
        We trained our model using the Adam optimizer with a learning rate of 0.001 
        and batch size of 32. The model was trained for 100 epochs on a single NVIDIA 
        V100 GPU. We used the CIFAR-10 dataset split into 45,000 training, 5,000 
        validation, and 10,000 test samples. Random seed was set to 42 for reproducibility.
        We applied dropout with rate 0.5 and weight decay of 1e-4. The model architecture 
        consists of 5 convolutional layers followed by 2 fully connected layers.
        """
    
    print("Analyzing methods section...")
    gaps, summary, report = analyze_paper(text)
    print(report)
