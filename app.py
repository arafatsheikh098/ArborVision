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
    page_title="ArborVision",
    page_icon="🌳",
    layout="wide",
)

st.title("🌳 ArborVision")
st.subheader("Computer Vision Benchmarking Platform")

st.markdown(
    """
    **ArborVision** lets you upload images, run multiple computer-vision
    models side-by-side, and compare their predictions and metrics
    (classification, instance segmentation, precision / recall / F1,
    inference time).

    The initial benchmark dataset is **tree imagery** — extensible to any
    other CV task via the model adapter interface.

    ---
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Version", ARBORVISION_VERSION)
with col2:
    st.metric("Device", get_device().upper())
with col3:
    st.metric("Available models", len(list_models()))

st.markdown("### Available models")
st.code(", ".join(list_models()) or "(none registered)")

st.markdown(
    """
    ### Where to start
    1. **Upload & Run** — pick one or more models and run them on your images.
    2. **Compare Results** — see side-by-side overlays, confusion matrix, and timing.
    3. **Export** — download a ZIP with all results and annotated images.

    The backend FastAPI server (run separately with `uvicorn`) does the
    heavy lifting; this UI is a thin client over its `/predict`,
    `/benchmark`, and `/experiments` endpoints.
    """
)

with st.expander("Backend connection"):
    api_url = st.text_input(
        "API base URL",
        value=st.session_state.get("api_url", "http://127.0.0.1:8765"),
        help="Where the FastAPI server is listening. Default: http://127.0.0.1:8765",
    )
    st.session_state["api_url"] = api_url
    st.caption(
        "Start the API with: `uvicorn arborvision.api.server:app --port 8765`"
    )