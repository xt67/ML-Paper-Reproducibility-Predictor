"""
Fix Hint Generator Module.
Generates actionable one-line suggestions for missing checklist items using LLM.
"""

import os
import time
from typing import Optional

import requests

# HuggingFace Inference API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Rate limiting
MAX_RETRIES = 3
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 30

# Prompt template
HINT_PROMPT_TEMPLATE = """You are helping an ML researcher improve their paper's reproducibility.

Missing checklist item ({severity} severity):
"{item}"

The paper's methods section does not adequately address this item.

Generate a single, actionable sentence (under 100 characters) that tells the author exactly what to add. Start with a verb.

Hint:"""


class HintGenerator:
    """
    Generates fix hints for missing reproducibility items using LLM inference.
    Falls back gracefully if no API token or API unavailable.
    """
    
    def __init__(self, api_token: Optional[str] = None, use_fallback: bool = True):
        """
        Initialize hint generator.
        
        Args:
            api_token: HuggingFace API token. Uses HF_TOKEN env var if None.
            use_fallback: Whether to use template-based hints if API fails.
        """
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.use_fallback = use_fallback
        self.headers = {}
        
        if self.api_token:
            self.headers = {"Authorization": f"Bearer {self.api_token}"}
            print("HintGenerator initialized with HF API token")
        else:
            print("HintGenerator: No API token found. Will use template-based hints.")
    
    def _call_api(self, prompt: str, api_url: str = HF_API_URL) -> Optional[str]:
        """
        Call HuggingFace Inference API.
        
        Args:
            prompt: The prompt to send
            api_url: API endpoint URL
            
        Returns:
            Generated text or None on failure
        """
        if not self.api_token:
            return None
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False,
            }
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get("generated_text", "")
                        # Clean up the response
                        text = text.strip().split("\n")[0]  # First line only
                        return text
                
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    wait_time = response.json().get("estimated_time", RETRY_DELAY)
                    print(f"Model loading, waiting {wait_time}s...")
                    time.sleep(min(wait_time, 30))
                    continue
                
                else:
                    print(f"API error {response.status_code}: {response.text[:100]}")
                    
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                
            time.sleep(RETRY_DELAY)
        
        return None
    
    def _get_template_hint(self, item: dict) -> str:
        """
        Generate a template-based hint when API is unavailable.
        Uses predefined hints based on category and keywords.
        """
        item_text = item.get("item", "").lower()
        category = item.get("category", "").lower()
        severity = item.get("severity", "medium")
        
        # Template hints by category/keywords
        templates = {
            # Hyperparameters
            "learning rate": "Specify the learning rate value (e.g., 'learning rate = 0.001').",
            "batch size": "State the batch size used for training (e.g., 'batch size = 32').",
            "optimizer": "Name the optimizer and its settings (e.g., 'Adam with β1=0.9').",
            "hyperparameters": "List all hyperparameter values in a table or dedicated section.",
            
            # Random seeds
            "random seed": "Report the random seed(s) used (e.g., 'seed = 42').",
            "seed": "Specify random seeds for initialization and data shuffling.",
            
            # Dataset
            "dataset": "State the exact dataset name and version used.",
            "split": "Specify train/validation/test split ratios and sample counts.",
            "preprocessing": "Describe data preprocessing steps (normalization, augmentation).",
            
            # Model
            "architecture": "Provide a complete model architecture description or diagram.",
            "parameters": "Report the total number of model parameters.",
            "activation": "Specify activation functions used in each layer.",
            "loss": "State the loss function used for training.",
            
            # Compute
            "gpu": "Specify hardware used (e.g., 'NVIDIA V100 GPU').",
            "training time": "Report total training time and wall-clock hours.",
            "memory": "State GPU memory requirements.",
            
            # Evaluation
            "metrics": "Define all evaluation metrics and how they are computed.",
            "baseline": "Compare against at least one relevant baseline method.",
            "error bars": "Report standard deviation or confidence intervals.",
            "runs": "State the number of independent runs performed.",
            
            # Code
            "code": "Provide a link to the source code repository.",
            "documentation": "Include a README with setup instructions.",
            "dependencies": "List software dependencies with version numbers.",
            
            # Other
            "limitations": "Add a limitations section discussing constraints.",
            "societal": "Discuss potential societal impacts of the work.",
            "pseudocode": "Include pseudocode for the main algorithm.",
        }
        
        # Find matching template
        for keyword, hint in templates.items():
            if keyword in item_text:
                return hint
        
        # Category-based fallback
        category_hints = {
            "experiments": f"Add missing experimental detail: {item.get('item', '')[:50]}...",
            "data": "Provide more details about the dataset and preprocessing.",
            "model": "Describe the model architecture in more detail.",
            "compute": "Specify computing resources and runtime.",
            "evaluation": "Add more evaluation details and comparisons.",
            "code": "Provide code access or implementation details.",
            "theory": "Include theoretical analysis or proofs.",
            "claims": "Support claims with evidence.",
            "limitations": "Discuss limitations of the approach.",
            "reproducibility": "Add details needed for reproducibility.",
        }
        
        if category in category_hints:
            return category_hints[category]
        
        # Generic fallback
        return f"Add: {item.get('item', 'missing detail')[:60]}..."
    
    def generate_hint(self, item: dict, context: str = "") -> str:
        """
        Generate a fix hint for a missing checklist item.
        
        Args:
            item: Checklist item dict with 'item', 'severity', 'category'
            context: Optional context from the paper (for more targeted hints)
            
        Returns:
            A single actionable hint string
        """
        # Try API first
        if self.api_token:
            prompt = HINT_PROMPT_TEMPLATE.format(
                severity=item.get("severity", "medium"),
                item=item.get("item", ""),
            )
            
            hint = self._call_api(prompt)
            if hint and len(hint) > 10:
                return hint
        
        # Fallback to template
        if self.use_fallback:
            return self._get_template_hint(item)
        
        return ""
    
    def generate_hints_batch(
        self, 
        missing_items: list[dict],
        context: str = "",
        max_items: int = 10,
    ) -> list[dict]:
        """
        Generate hints for multiple missing items.
        
        Args:
            missing_items: List of missing checklist items
            context: Optional context from the paper
            max_items: Maximum items to process (to limit API calls)
            
        Returns:
            List of items with 'hint' field added
        """
        # Sort by severity (high first) and limit
        severity_order = {"high": 0, "medium": 1, "low": 2}
        items = sorted(
            missing_items,
            key=lambda x: severity_order.get(x.get("severity", "medium"), 1)
        )[:max_items]
        
        results = []
        for item in items:
            hint = self.generate_hint(item, context)
            item_with_hint = item.copy()
            item_with_hint["hint"] = hint
            results.append(item_with_hint)
        
        return results
    
    def format_hints_report(self, items_with_hints: list[dict]) -> str:
        """Format hints as a readable report."""
        lines = [
            "=" * 60,
            "FIX SUGGESTIONS",
            "=" * 60,
            "",
        ]
        
        current_severity = None
        for item in items_with_hints:
            if item.get("severity") != current_severity:
                current_severity = item.get("severity")
                lines.append(f"\n[{current_severity.upper()} PRIORITY]")
            
            lines.append(f"\n• Missing: {item.get('item', '')[:60]}...")
            lines.append(f"  → Fix: {item.get('hint', 'No hint available')}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def generate_hints_for_gaps(
    gaps: list[dict],
    context: str = "",
    api_token: Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Convenience function to generate hints for gap detection results.
    
    Args:
        gaps: Output from GapDetector.detect()
        context: Optional paper context
        api_token: Optional HF API token
        
    Returns:
        Tuple of (items_with_hints, formatted_report)
    """
    # Filter to missing items only
    missing = [g for g in gaps if g.get("status") == "missing"]
    
    generator = HintGenerator(api_token=api_token)
    items_with_hints = generator.generate_hints_batch(missing, context)
    report = generator.format_hints_report(items_with_hints)
    
    return items_with_hints, report


if __name__ == "__main__":
    # Test hint generation
    sample_items = [
        {
            "id": 11,
            "item": "The paper specifies the random seeds used for initialization and data shuffling.",
            "severity": "high",
            "category": "experiments",
        },
        {
            "id": 10,
            "item": "The paper specifies all hyperparameters used, including learning rate, batch size, and optimizer settings.",
            "severity": "high",
            "category": "experiments",
        },
        {
            "id": 35,
            "item": "The paper provides access to the source code.",
            "severity": "high",
            "category": "code",
        },
    ]
    
    print("Testing hint generation...\n")
    
    generator = HintGenerator()
    items_with_hints = generator.generate_hints_batch(sample_items)
    report = generator.format_hints_report(items_with_hints)
    
    print(report)
