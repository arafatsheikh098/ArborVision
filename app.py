"""ArborVision — landing page.

Run with::

    streamlit run app.py

The other pages (Upload & Run, Compare Results, Export) live under the
``pages/`` directory and are auto-discovered by Streamlit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when launched via ``streamlit run app.py``
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from arborvision import __version__ as ARBORVISION_VERSION
from arborvision.config import get_device
from arborvision.models import list_models


st.set_page_config(
    page_title="ArborVision — CV Benchmark Platform",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling for ArborVision
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

      html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      
      .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.03) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.03) 0%, transparent 40%);
      }

      .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
      }

      .hero-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 1.5rem;
      }

      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 999px;
        font-size: 0.825rem;
        font-weight: 600;
        margin-bottom: 1rem;
      }

      div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease;
      }

      div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.4);
      }

      .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        transition: all 0.25s ease;
      }

      .feature-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateY(-3px);
      }

      .model-chip {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(59, 130, 246, 0.12);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="status-pill">⚡ Neural Vision Engine Active</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🌳 ArborVision</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">High-Precision Computer Vision Benchmarking Platform</div>', unsafe_allow_html=True)

st.markdown(
    """
    **ArborVision** allows computer vision engineers and researchers to benchmark and compare
    multiple vision models side-by-side on identical image sets. Inspect classification distributions,
    segmentation boundary overlays, and precision/recall/F1 matrices in real time.
    """
)

st.write("")

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Engine Version", ARBORVISION_VERSION)
with col2:
    st.metric("Compute Accelerator", get_device().upper())
with col3:
    st.metric("Benchmarked Models", len(list_models()))

st.write("")

# Available Models Section
st.markdown("### 🧬 Registered Vision Models")
models = list_models()
if models:
    chips_html = "".join([f'<span class="model-chip">{m}</span>' for m in models])
    st.markdown(chips_html, unsafe_allow_html=True)
else:
    st.info("No models registered yet.")

st.write("")
st.markdown("### 🚀 Interactive Workflows")

card_col1, card_col2, card_col3 = st.columns(3)

with card_col1:
    st.markdown(
        """
        <div class="feature-card">
          <h4>📤 1. Upload & Run</h4>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Upload tree canopy imagery or test sets, select benchmark backbones, and execute parallel batch inference.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_col2:
    st.markdown(
        """
        <div class="feature-card">
          <h4>⚖️ 2. Compare Results</h4>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Inspect side-by-side segmentation overlays, alpha blended masks, confidence distributions, and latency metrics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_col3:
    st.markdown(
        """
        <div class="feature-card">
          <h4>💾 3. Export Package</h4>
          <p style="color: #94a3b8; font-size: 0.9rem;">
            Export standardized JSON metrics, CSV timing datasets, and annotated mask images for research publications.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

with st.expander("⚙️ Backend API Configuration"):
    api_url = st.text_input(
        "FastAPI Service Endpoint",
        value=st.session_state.get("api_url", "http://127.0.0.1:8765"),
        help="Where the FastAPI server is listening. Default: http://127.0.0.1:8765",
    )
    st.session_state["api_url"] = api_url
    st.caption("Start the backend server using: `uvicorn arborvision.api.server:app --port 8765`")