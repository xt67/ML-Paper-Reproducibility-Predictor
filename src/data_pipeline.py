"""
Data pipeline for ML Paper Reproducibility Predictor.
Downloads Papers With Code data, fetches arXiv PDFs, extracts methods sections,
and creates train/val/test splits.
"""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from .pdf_extractor import extract_from_arxiv, _clean_extracted_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FAILED_LOG = DATA_DIR / "failed_extractions.txt"

# API settings
ARXIV_DELAY = 1.0  # Seconds between arXiv requests
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30


def load_pwc_annotations(filepath: str) -> pd.DataFrame:
    """
    Load Papers With Code reproducibility annotations.
    
    Args:
        filepath: Path to the PwC JSON/CSV file
        
    Returns:
        DataFrame with columns: paper_id, arxiv_id, label (0 or 1), title
    """
    path = Path(filepath)
    
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Convert to DataFrame
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data.get("papers", data))
    
    # Standardize column names
    column_mapping = {
        "paper_id": "paper_id",
        "id": "paper_id",
        "arxiv_id": "arxiv_id",
        "arxiv": "arxiv_id",
        "reproducibility_score": "label",
        "reproducible": "label",
        "is_reproducible": "label",
        "title": "title",
        "paper_title": "title",
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Ensure required columns exist
    required = ["arxiv_id", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert label to binary if needed
    if df["label"].dtype == "object" or df["label"].max() > 1:
        df["label"] = df["label"].apply(lambda x: 1 if x in [1, "1", True, "yes", "reproducible"] else 0)
    
    # Filter to papers with arXiv IDs
    df = df[df["arxiv_id"].notna() & (df["arxiv_id"] != "")]
    
    logger.info(f"Loaded {len(df)} papers with arXiv IDs")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def fetch_arxiv_pdf(arxiv_id: str, save_dir: str) -> Optional[str]:
    """
    Download PDF from arXiv.
    
    Args:
        arxiv_id: arXiv paper ID
        save_dir: Directory to save PDF
        
    Returns:
        Local filepath if successful, None otherwise
    """
    save_path = Path(save_dir) / f"{arxiv_id.replace('/', '_')}.pdf"
    
    # Skip if already exists
    if save_path.exists():
        return str(save_path)
    
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Verify it's actually a PDF
            if not response.content.startswith(b"%PDF"):
                logger.warning(f"Not a valid PDF for {arxiv_id}")
                return None
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(response.content)
            
            return str(save_path)
            
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {arxiv_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(ARXIV_DELAY * (attempt + 1))
    
    return None


def extract_methods_section(pdf_path: str) -> Optional[str]:
    """
    Extract methods section from a PDF file.
    Wrapper around pdf_extractor with error handling.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted methods text, or None on failure
    """
    try:
        from .pdf_extractor import extract_from_pdf
        text = extract_from_pdf(pdf_path, max_tokens=2000)
        
        if len(text) < 100:
            logger.warning(f"Extraction too short ({len(text)} chars) for {pdf_path}")
            return None
            
        return text
        
    except Exception as e:
        logger.error(f"Extraction failed for {pdf_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """
    Clean extracted text for model input.
    Delegates to pdf_extractor._clean_extracted_text.
    """
    return _clean_extracted_text(text)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    test_ratio: float = 0.125,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split maintaining class balance in each split.
    
    Args:
        df: DataFrame with 'label' column
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=seed
    )
    
    # Second split: val vs test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df["label"],
        random_state=seed
    )
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split["label"].value_counts().to_dict()
        logger.info(f"  {name} label distribution: {dist}")
    
    return train_df, val_df, test_df


def run_pipeline(
    annotations_path: Optional[str] = None,
    skip_existing: bool = True,
    max_papers: Optional[int] = None
):
    """
    Run the full data pipeline.
    
    1. Load PwC annotations
    2. For each paper, fetch PDF and extract methods section
    3. Clean text
    4. Split into train/val/test
    5. Save as Parquet
    
    Args:
        annotations_path: Path to annotations file. If None, uses default.
        skip_existing: Skip papers where PDF already exists
        max_papers: Limit number of papers to process (for testing)
    """
    # Setup directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    if annotations_path is None:
        # Look for any annotation file in raw directory
        possible_files = list(RAW_DIR.glob("*.json*")) + list(RAW_DIR.glob("*.csv"))
        if not possible_files:
            logger.error("No annotation file found in data/raw/. Please download first.")
            logger.info("Run: python scripts/download_data.py")
            return
        annotations_path = str(possible_files[0])
    
    logger.info(f"Loading annotations from {annotations_path}")
    df = load_pwc_annotations(annotations_path)
    
    if max_papers:
        df = df.head(max_papers)
        logger.info(f"Limited to {max_papers} papers for testing")
    
    # Track failures
    failed = []
    
    # Extract methods for each paper
    methods_texts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
        arxiv_id = row["arxiv_id"]
        
        # Fetch PDF
        pdf_path = fetch_arxiv_pdf(arxiv_id, str(RAW_DIR))
        
        if pdf_path is None:
            failed.append((arxiv_id, "PDF download failed"))
            methods_texts.append(None)
            continue
        
        # Rate limiting
        time.sleep(ARXIV_DELAY)
        
        # Extract methods
        methods_text = extract_methods_section(pdf_path)
        
        if methods_text is None:
            failed.append((arxiv_id, "Methods extraction failed"))
            methods_texts.append(None)
            continue
        
        methods_texts.append(methods_text)
    
    # Add methods text to DataFrame
    df["methods_text"] = methods_texts
    
    # Remove failed extractions
    valid_df = df[df["methods_text"].notna()].copy()
    logger.info(f"Successfully processed {len(valid_df)}/{len(df)} papers")
    
    # Log failures
    if failed:
        with open(FAILED_LOG, "w") as f:
            for arxiv_id, reason in failed:
                f.write(f"{arxiv_id}\t{reason}\n")
        logger.info(f"Logged {len(failed)} failures to {FAILED_LOG}")
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(valid_df)
    
    # Save as Parquet
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    
    logger.info(f"Saved splits to {PROCESSED_DIR}")
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--annotations", type=str, help="Path to annotations file")
    parser.add_argument("--max-papers", type=int, help="Limit papers to process")
    parser.add_argument("--no-skip", action="store_true", help="Re-download existing PDFs")
    
    args = parser.parse_args()
    
    run_pipeline(
        annotations_path=args.annotations,
        skip_existing=not args.no_skip,
        max_papers=args.max_papers
    )
