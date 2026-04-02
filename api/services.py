"""
Analysis service that orchestrates all ML components.
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from src.classifier import ReproducibilityClassifier
from src.explainer import SHAPExplainer
from src.gap_detector import GapDetector
from src.hint_generator import HintGenerator
from src.pdf_extractor import extract_from_arxiv, extract_from_pdf, extract_from_url


class AnalysisService:
    """
    Orchestrates the full analysis pipeline:
    1. PDF extraction
    2. Classification
    3. Gap detection
    4. SHAP explanation
    5. Hint generation
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        checklist_path: Optional[str] = None,
    ):
        """
        Initialize analysis service with all components.
        
        Args:
            model_path: Path to fine-tuned SciBERT model.
                       If None, uses pretrained model.
            checklist_path: Path to checklist JSON.
                           If None, uses default.
        """
        # Try to load fine-tuned model, fall back to pretrained
        default_model_path = Path("models/scibert_finetuned")
        if model_path:
            self.model_path = model_path
        elif default_model_path.exists():
            self.model_path = str(default_model_path)
        else:
            self.model_path = None
        
        print(f"Initializing AnalysisService...")
        print(f"  Model path: {self.model_path or 'pretrained'}")
        
        # Initialize classifier
        self.classifier = ReproducibilityClassifier(self.model_path)
        
        # Initialize gap detector
        print("  Loading gap detector...")
        self.gap_detector = GapDetector(checklist_path=checklist_path)
        
        # Initialize SHAP explainer
        print("  Initializing SHAP explainer...")
        self.explainer = SHAPExplainer(self.classifier, cache_dir=".cache/shap")
        
        # Initialize hint generator
        print("  Initializing hint generator...")
        self.hint_generator = HintGenerator(use_fallback=True)
        
        print("AnalysisService ready!")
    
    def extract_from_arxiv(self, arxiv_id: str, max_tokens: int = 2000) -> str:
        """Extract methods section from arXiv paper."""
        return extract_from_arxiv(arxiv_id, max_tokens)
    
    def extract_from_url(self, url: str, max_tokens: int = 2000) -> str:
        """Extract methods section from PDF URL."""
        return extract_from_url(url, max_tokens)
    
    async def extract_from_upload(
        self, 
        file: UploadFile, 
        max_tokens: int = 2000
    ) -> str:
        """Extract methods section from uploaded PDF."""
        # Save to temp file
        content = await file.read()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            return extract_from_pdf(tmp_path, max_tokens)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def analyze(self, methods_text: str) -> dict:
        """
        Run full analysis pipeline.
        
        Args:
            methods_text: Extracted methods section text
            
        Returns:
            Dictionary with classification, gaps, explanation, and hints
        """
        # 1. Classification
        classification = self.classifier.predict(methods_text)
        
        # 2. Gap detection
        gaps = self.gap_detector.detect(methods_text)
        gap_summary = self.gap_detector.summary(gaps)
        
        # 3. SHAP explanation
        explanation = self.explainer.explain(methods_text, top_k=5)
        
        # 4. Generate hints for missing items
        missing_items = [g for g in gaps if g["status"] == "missing"]
        items_with_hints = self.hint_generator.generate_hints_batch(
            missing_items,
            context=methods_text,
            max_items=10,  # Limit API calls
        )
        
        # Merge hints back into gaps
        hint_map = {item["id"]: item.get("hint", "") for item in items_with_hints}
        for gap in gaps:
            gap["hint"] = hint_map.get(gap["id"], "")
        
        return {
            "classification": classification,
            "gaps": gaps,
            "gap_summary": gap_summary,
            "explanation": explanation,
        }
