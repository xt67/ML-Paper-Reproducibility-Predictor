"""
PDF text extraction module for ML papers.
Extracts methods/methodology section from academic PDFs.
"""

import re
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import requests


# Regex patterns for detecting methods section headers
SECTION_HEADERS = [
    r"^\s*\d*\.?\s*(methods?|methodology)\s*$",
    r"^\s*\d*\.?\s*(approach|proposed method|our method)\s*$",
    r"^\s*\d*\.?\s*(model|architecture|system design)\s*$",
    r"^\s*\d*\.?\s*(technical approach|framework)\s*$",
]

# Markers indicating end of methods section
END_MARKERS = [
    r"^\s*\d*\.?\s*(results?|experiments?|evaluation)\s*$",
    r"^\s*\d*\.?\s*(empirical|analysis)\s*$",
    r"^\s*\d*\.?\s*(conclusion|discussion|related work)\s*$",
    r"^\s*\d*\.?\s*(references?|bibliography|appendix)\s*$",
]

# Compile patterns for efficiency
SECTION_PATTERN = re.compile("|".join(SECTION_HEADERS), re.IGNORECASE | re.MULTILINE)
END_PATTERN = re.compile("|".join(END_MARKERS), re.IGNORECASE | re.MULTILINE)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract full text from PDF using PyMuPDF.
    Preserves paragraph structure by joining pages with double newlines.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Full text content of the PDF
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)
    
    doc.close()
    return "\n\n".join(pages_text)


def find_methods_section(full_text: str, max_tokens: int = 2000) -> str:
    """
    Find and extract the methods/methodology section from paper text.
    
    Args:
        full_text: Complete text of the paper
        max_tokens: Maximum characters to return (rough token approximation)
        
    Returns:
        Methods section text, or fallback text if not found
    """
    # Try to find methods section header
    match = SECTION_PATTERN.search(full_text)
    
    if match:
        start_idx = match.start()
        
        # Find the end of methods section
        remaining_text = full_text[match.end():]
        end_match = END_PATTERN.search(remaining_text)
        
        if end_match:
            methods_text = full_text[start_idx:match.end() + end_match.start()]
        else:
            # No end marker found, take reasonable chunk
            methods_text = full_text[start_idx:start_idx + max_tokens * 4]
    else:
        # Fallback: skip abstract (usually first ~500 chars) and take body
        # Try to find "Abstract" or "Introduction" to skip
        intro_match = re.search(
            r"^\s*\d*\.?\s*(introduction|1\.\s*introduction)", 
            full_text, 
            re.IGNORECASE | re.MULTILINE
        )
        
        if intro_match:
            start_idx = intro_match.start()
        else:
            # Skip first 500 chars as likely abstract
            start_idx = min(500, len(full_text) // 10)
        
        methods_text = full_text[start_idx:]
    
    # Clean and truncate
    methods_text = _clean_extracted_text(methods_text)
    
    # Truncate to max_tokens (rough char approximation: 1 token ≈ 4 chars)
    if len(methods_text) > max_tokens * 4:
        methods_text = methods_text[:max_tokens * 4]
        # Try to end at sentence boundary
        last_period = methods_text.rfind(".")
        if last_period > max_tokens * 2:
            methods_text = methods_text[:last_period + 1]
    
    return methods_text


def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted PDF text.
    
    - Remove LaTeX commands
    - Remove URLs
    - Normalize whitespace
    - Remove figure/table captions
    - Remove page numbers
    """
    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)  # \command{...}
    text = re.sub(r"\\[a-zA-Z]+", "", text)  # \command
    text = re.sub(r"\$[^$]+\$", " [MATH] ", text)  # $...$
    text = re.sub(r"\$\$[^$]+\$\$", " [MATH] ", text)  # $$...$$
    
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    
    # Remove figure/table captions
    text = re.sub(r"Figure\s*\d+[.:][^\n]*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Table\s*\d+[.:][^\n]*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Fig\.\s*\d+[.:][^\n]*\n", "", text, flags=re.IGNORECASE)
    
    # Remove lines that are just numbers (page numbers, table data)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = text.strip()
    
    return text


def extract_from_pdf(pdf_path: str, max_tokens: int = 2000) -> str:
    """
    Main extraction function: extract methods section from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        max_tokens: Maximum tokens to extract
        
    Returns:
        Cleaned methods section text
    """
    full_text = extract_text_from_pdf(pdf_path)
    return find_methods_section(full_text, max_tokens)


def extract_from_url(url: str, max_tokens: int = 2000, timeout: int = 30) -> str:
    """
    Download PDF from URL and extract methods section.
    Used by the API for live URL input.
    
    Args:
        url: URL to PDF file (e.g., arxiv.org/pdf/...)
        max_tokens: Maximum tokens to extract
        timeout: Request timeout in seconds
        
    Returns:
        Cleaned methods section text
    """
    # Download to temp file
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    try:
        return extract_from_pdf(tmp_path, max_tokens)
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def extract_from_arxiv(arxiv_id: str, max_tokens: int = 2000) -> str:
    """
    Extract methods section from arXiv paper by ID.
    
    Args:
        arxiv_id: arXiv paper ID (e.g., "2301.00001" or "2301.00001v1")
        max_tokens: Maximum tokens to extract
        
    Returns:
        Cleaned methods section text
    """
    # Normalize arxiv_id (remove version if present for URL)
    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    return extract_from_url(url, max_tokens)


if __name__ == "__main__":
    # Test with a sample arXiv paper
    import sys
    
    if len(sys.argv) > 1:
        arxiv_id = sys.argv[1]
        print(f"Extracting methods from arXiv:{arxiv_id}")
        text = extract_from_arxiv(arxiv_id)
        print(f"\n--- Extracted {len(text)} characters ---\n")
        print(text[:2000])
    else:
        print("Usage: python pdf_extractor.py <arxiv_id>")
        print("Example: python pdf_extractor.py 2301.00001")
