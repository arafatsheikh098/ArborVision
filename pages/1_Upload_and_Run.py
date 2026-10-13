"""Page 1 — Upload images, pick models, run predictions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from arborvision.config import DEFAULT_MASK_THRESHOLD, DEFAULT_SCORE_THRESHOLD
from arborvision.models import list_models
from arborvision.ui.client import (
    fallback_local_predict,
    get,
    is_api_alive,
    post_predict,
)


st.set_page_config(page_title="Upload & Run — ArborVision", page_icon="📤", layout="wide")
st.title("📤 Upload & Run")

# Inline backend URL editor so the user doesn't have to go back to the
# home page to fix a wrong port. This is the most common cause of "error"
# on this page.
from arborvision.ui.client import api_url as _current_api_url
with st.expander("⚙️ Backend settings (click to expand if you see errors)"):
    st.caption(f"Current API URL: **{_current_api_url()}**")
    new_url = st.text_input(
        "API base URL",
        value=st.session_state.get("api_url", "http://127.0.0.1:8765"),
        key="api_url_editor",
        help="If the UI shows 🟡 yellow dot, the API isn't reachable. Fix the URL here.",
    )
    if new_url != st.session_state.get("api_url"):
        st.session_state["api_url"] = new_url
        st.rerun()

# Pull model list (try API first, fall back to registry)
models_response = get("/models")
api_models = [m["name"] for m in models_response.get("models", [])] if models_response else []
local_models = list_models()
available = sorted(set(api_models) | set(local_models))

if not available:
    st.error(
        "**No models are registered.** \n\n"
        f"- Local registry returned: `{local_models}` \n"
        f"- API at `st.session_state['api_url']` returned: `{models_response}` \n\n"
        "**Fix:** Scroll to the bottom of the home page, expand **Backend connection**, "
        "and set the API URL to the right port. Or restart the API with "
        "`uvicorn arborvision.api.server:app --port 8765`."
    )
    st.stop()

st.sidebar.subheader("Backend")
api_alive = is_api_alive()
st.sidebar.write(
    "🟢 API connected" if api_alive else "🟡 Using local fallback (API unreachable)"
)

with st.form("run_form"):
    st.markdown("### 1. Choose models")
    selected = st.multiselect(
        "Models to run",
        options=available,
        default=available[: min(2, len(available))],
        help="Hold ⌘/Ctrl to select multiple.",
    )

    st.markdown("### 2. Detection thresholds (instance segmentation)")
    score_threshold = st.slider(
        "Score threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SCORE_THRESHOLD,
        step=0.05,
    )
    mask_threshold = st.slider(
        "Mask threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MASK_THRESHOLD,
        step=0.05,
    )

    st.markdown("### 3. Upload images")
    files = st.file_uploader(
        "One or more image files",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    submitted = st.form_submit_button("Run benchmark", use_container_width=True)

if submitted:
    if not selected:
        st.error("Pick at least one model.")
        st.stop()
    if not files:
        st.error("Upload at least one image.")
        st.stop()

    progress = st.progress(0.0, text="Running…")
    result = None
    if api_alive:
        result = post_predict(files, selected)
    else:
        # Local fallback: process images one by one and merge into the
        # same shape as the API response.
        agg: dict[str, list[dict]] = {n: [] for n in selected}
        total = 0.0
        for i, f in enumerate(files):
            r = fallback_local_predict(f.getvalue(), selected)
            for n in selected:
                agg[n].extend(r["predictions"].get(n, []))
            total += r["total_latency_ms"]
            progress.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}")
        result = {"predictions": agg, "total_latency_ms": total}
    progress.progress(1.0, text="Done.")

    if result and result.get("predictions"):
        st.session_state["last_predictions"] = result
        st.session_state["last_files"] = [
            {"name": f.name, "bytes": f.getvalue()} for f in files
        ]
        st.session_state["last_models"] = selected
        st.session_state["last_thresholds"] = {
            "score_threshold": score_threshold,
            "mask_threshold": mask_threshold,
        }
        st.success(
            f"Done. Total latency: {result['total_latency_ms']:.1f} ms across {len(files)} image(s)."
        )
        st.info("Open **Compare Results** in the sidebar to see overlays and metrics.")
    else:
        st.error("No predictions returned. Check the backend logs.")

# Quick view of the last run's summary
last = st.session_state.get("last_predictions")
if last:
    st.markdown("### Last run summary")
    for model_name, preds in last["predictions"].items():
        with st.expander(f"Model: {model_name} — {len(preds)} image(s)"):
            for i, p in enumerate(preds):
                st.write(
                    f"  - image {i + 1}: latency={p['latency_ms']:.1f}ms, "
                    f"top-1={p.get('labels', [''])[0] or p.get('topk_labels', [''])[0]!r}, "
                    f"detections={p.get('n_masks', len(p.get('labels', [])))}"
                )