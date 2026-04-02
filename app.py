"""
Streamlit Frontend for ML Paper Reproducibility Predictor.

A web interface to analyze ML papers for reproducibility.
"""

import sys
import os
from pathlib import Path

# Download ALL NLTK data before importing modules that need it
import nltk

# Set NLTK data path to a writable location
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download all required resources
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
    except Exception:
        pass

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import ReproducibilityClassifier
from src.explainer import SHAPExplainer
from src.gap_detector import GapDetector
from src.hint_generator import HintGenerator
from src.pdf_extractor import extract_from_arxiv, extract_from_pdf, extract_from_url

# Page configuration
st.set_page_config(
    page_title="ML Reproducibility Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Score gauge styling */
    .score-container {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .score-high {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 2px solid #ffc107;
    }
    
    .score-low {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
    }
    
    .score-label {
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    
    /* Gap table styling */
    .gap-high {
        background-color: #f8d7da !important;
    }
    
    .gap-medium {
        background-color: #fff3cd !important;
    }
    
    .gap-low {
        background-color: #d1ecf1 !important;
    }
    
    /* Highlighted text */
    .highlight-green {
        background-color: #d4edda;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    .highlight-yellow {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .score-value {
            font-size: 3rem;
        }
        .score-label {
            font-size: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load ML models (cached for performance)."""
    model_path = Path("models/scibert_finetuned")
    if model_path.exists():
        classifier = ReproducibilityClassifier(str(model_path))
    else:
        classifier = ReproducibilityClassifier()
    
    gap_detector = GapDetector()
    explainer = SHAPExplainer(classifier, cache_dir=".cache/shap")
    hint_generator = HintGenerator(use_fallback=True)
    
    return classifier, gap_detector, explainer, hint_generator


def render_score_gauge(score: float, label: str):
    """Render the reproducibility score as a gauge visualization."""
    # Determine color class based on score
    if score >= 0.7:
        color_class = "score-high"
        emoji = "✅"
    elif score >= 0.4:
        color_class = "score-medium"
        emoji = "⚠️"
    else:
        color_class = "score-low"
        emoji = "❌"
    
    # Create gauge HTML
    percentage = int(score * 100)
    st.markdown(f"""
    <div class="score-container {color_class}">
        <p class="score-value">{percentage}%</p>
        <p class="score-label">{emoji} {label}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar visualization
    st.progress(score)


def render_gap_table(gaps: list, gap_summary: dict):
    """Render the gap report table sorted by severity."""
    st.subheader("📋 Gap Report")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coverage", f"{gap_summary['coverage_score']:.1f}%")
    with col2:
        st.metric("Missing Items", gap_summary['missing'])
    with col3:
        st.metric("High Severity", gap_summary['missing_high_severity'], delta_color="inverse")
    with col4:
        st.metric("Present Items", gap_summary['present'])
    
    # Filter to missing items
    missing_gaps = [g for g in gaps if g["status"] == "missing"]
    
    if not missing_gaps:
        st.success("🎉 No critical gaps detected! The paper addresses all checklist items.")
        return
    
    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    missing_gaps.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["id"]))
    
    # Create expandable sections by severity
    high_gaps = [g for g in missing_gaps if g["severity"] == "high"]
    medium_gaps = [g for g in missing_gaps if g["severity"] == "medium"]
    low_gaps = [g for g in missing_gaps if g["severity"] == "low"]
    
    if high_gaps:
        with st.expander(f"🔴 High Severity ({len(high_gaps)} items)", expanded=True):
            for gap in high_gaps:
                render_gap_item(gap)
    
    if medium_gaps:
        with st.expander(f"🟡 Medium Severity ({len(medium_gaps)} items)", expanded=True):
            for gap in medium_gaps:
                render_gap_item(gap)
    
    if low_gaps:
        with st.expander(f"🔵 Low Severity ({len(low_gaps)} items)", expanded=False):
            for gap in low_gaps:
                render_gap_item(gap)


def render_gap_item(gap: dict):
    """Render a single gap item with hint."""
    st.markdown(f"**{gap['item'][:100]}{'...' if len(gap['item']) > 100 else ''}**")
    
    if gap.get("hint"):
        st.info(f"💡 **Fix:** {gap['hint']}")
    
    st.caption(f"Category: {gap['category']} | Similarity: {gap['similarity_score']:.2f}")
    st.divider()


def render_highlighted_text(explanation: dict):
    """Render methods text with SHAP-based sentence coloring."""
    st.subheader("🔍 Explanation: Key Sentences")
    
    st.caption("Sentences are colored by their influence on the reproducibility score:")
    st.markdown("🟢 **Green** = Increases score | 🟡 **Yellow** = Decreases score | ⚪ Neutral")
    
    # Render highlighted text
    highlighted_html = ""
    for segment in explanation.get("highlighted_text", []):
        text = segment["text"]
        color = segment["color"]
        
        if color == "green":
            highlighted_html += f'<span class="highlight-green">{text}</span> '
        elif color == "yellow":
            highlighted_html += f'<span class="highlight-yellow">{text}</span> '
        else:
            highlighted_html += f'{text} '
    
    st.markdown(f"<p>{highlighted_html}</p>", unsafe_allow_html=True)
    
    # Top influential sentences
    st.markdown("**Top Influential Sentences:**")
    for sent in explanation.get("sentences", [])[:5]:
        score = sent["normalized_score"]
        emoji = "🟢" if score > 0 else "🟡" if score < 0 else "⚪"
        sign = "+" if score > 0 else ""
        st.markdown(f"{emoji} ({sign}{score:.2f}) *{sent['sentence'][:100]}{'...' if len(sent['sentence']) > 100 else ''}*")


def analyze_paper(methods_text: str, classifier, gap_detector, explainer, hint_generator):
    """Run the full analysis pipeline."""
    # Classification
    classification = classifier.predict(methods_text)
    
    # Gap detection
    gaps = gap_detector.detect(methods_text)
    gap_summary = gap_detector.summary(gaps)
    
    # SHAP explanation
    explanation = explainer.explain(methods_text, top_k=5)
    
    # Generate hints for missing items
    missing_items = [g for g in gaps if g["status"] == "missing"]
    items_with_hints = hint_generator.generate_hints_batch(
        missing_items,
        context=methods_text,
        max_items=10,
    )
    
    # Merge hints back
    hint_map = {item["id"]: item.get("hint", "") for item in items_with_hints}
    for gap in gaps:
        gap["hint"] = hint_map.get(gap["id"], "")
    
    return classification, gaps, gap_summary, explanation


def main():
    """Main Streamlit app."""
    # Header
    st.title("🔬 ML Paper Reproducibility Predictor")
    st.markdown("""
    Analyze your ML paper's methods section for reproducibility issues.
    Upload a PDF, enter an arXiv ID, or paste text directly.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📤 Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload PDF", "🔗 arXiv ID", "📝 Paste Text"],
            index=2,
        )
        
        st.divider()
        st.header("ℹ️ About")
        st.markdown("""
        This tool analyzes ML papers for reproducibility by:
        
        1. **Scoring** reproducibility (0-100%)
        2. **Detecting gaps** against NeurIPS checklist
        3. **Explaining** which sentences matter
        4. **Suggesting** fixes for issues
        
        Built with SciBERT, sentence-transformers, and SHAP.
        """)
    
    # Load models
    with st.spinner("Loading models... (first time may take a minute)"):
        classifier, gap_detector, explainer, hint_generator = load_models()
    
    # Input section
    methods_text = None
    
    if input_method == "📄 Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help="Upload an ML paper PDF. We'll extract the methods section.",
        )
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    methods_text = extract_from_pdf(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
    
    elif input_method == "🔗 arXiv ID":
        arxiv_id = st.text_input(
            "Enter arXiv ID",
            placeholder="e.g., 2301.00001",
            help="Enter the arXiv paper ID (e.g., 2301.00001 or 2301.00001v2)",
        )
        
        if arxiv_id:
            with st.spinner(f"Fetching paper from arXiv: {arxiv_id}..."):
                try:
                    methods_text = extract_from_arxiv(arxiv_id)
                except Exception as e:
                    st.error(f"Failed to fetch from arXiv: {e}")
    
    else:  # Paste Text
        methods_text = st.text_area(
            "Paste your methods section",
            height=200,
            placeholder="Paste the methods/methodology section of your paper here...",
            help="Paste at least 50 characters of text from your paper's methods section.",
        )
    
    # Show extracted text preview
    if methods_text and len(methods_text.strip()) >= 50:
        with st.expander("📖 Extracted Text Preview", expanded=False):
            st.text(methods_text[:1000] + ("..." if len(methods_text) > 1000 else ""))
        
        # Analyze button
        if st.button("🚀 Analyze Reproducibility", type="primary", use_container_width=True):
            with st.spinner("Analyzing paper... This may take 10-30 seconds."):
                try:
                    classification, gaps, gap_summary, explanation = analyze_paper(
                        methods_text, classifier, gap_detector, explainer, hint_generator
                    )
                    
                    # Store results in session state
                    st.session_state["results"] = {
                        "classification": classification,
                        "gaps": gaps,
                        "gap_summary": gap_summary,
                        "explanation": explanation,
                    }
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return
    
    elif methods_text and len(methods_text.strip()) < 50:
        st.warning("Please enter at least 50 characters of text.")
    
    # Display results
    if "results" in st.session_state:
        results = st.session_state["results"]
        
        st.divider()
        st.header("📊 Analysis Results")
        
        # Score gauge
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score = results["classification"]["score"]
            label = "Reproducible" if results["classification"]["label"] == 1 else "Not Reproducible"
            render_score_gauge(score, label)
            
            st.caption(f"Confidence: {results['classification']['confidence']:.1%}")
        
        with col2:
            # Gap summary
            render_gap_table(results["gaps"], results["gap_summary"])
        
        st.divider()
        
        # SHAP explanation
        render_highlighted_text(results["explanation"])
        
        # Download results button
        st.divider()
        import json
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=results_json,
            file_name="reproducibility_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
