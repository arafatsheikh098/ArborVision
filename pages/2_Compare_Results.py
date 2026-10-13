"""Page 2 — Side-by-side comparison of predictions, metrics, and timing."""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

from arborvision.data.loader import decode_upload
from arborvision.data.overlays import draw_prediction
from arborvision.ui.client import render_json


st.set_page_config(page_title="Compare Results — ArborVision", page_icon="📊", layout="wide")
st.title("📊 Compare Results")

last = st.session_state.get("last_predictions")
files = st.session_state.get("last_files", [])
models = st.session_state.get("last_models", [])
thresholds = st.session_state.get("last_thresholds", {"score_threshold": 0.5, "mask_threshold": 0.5})

if not last or not files:
    st.info("No results yet. Run predictions on the **Upload & Run** page first.")
    st.stop()

# Image picker
file_names = [f["name"] for f in files]
selected_name = st.selectbox("Image", file_names)
selected = next(f for f in files if f["name"] == selected_name)
image = decode_upload(selected["bytes"])

# Recompute overlays client-side so we always have something to display
# even if the API didn't return masks (e.g. classification-only model).
st.subheader("Overlays")
cols = st.columns(max(len(models), 1))
for ci, mname in enumerate(models):
    with cols[ci]:
        st.markdown(f"**{mname}**")
        preds = last["predictions"].get(mname, [])
        pred = next((p for p in preds if p.get("image_index", i) is not None or True), None) if preds else None
        # The API doesn't tag predictions with image indices, so we
        # align by order in the result list with the order of uploaded
        # files. We trust the order the API received them in.
        try:
            idx = file_names.index(selected_name)
            pred = preds[idx] if idx < len(preds) else None
        except Exception:
            pred = preds[0] if preds else None

        if pred is None:
            st.warning("No prediction for this image.")
            continue

        boxes = [tuple(b) for b in pred.get("boxes", [])]
        labels = pred.get("labels", []) or pred.get("topk_labels", [])
        scores = pred.get("scores", []) or pred.get("topk_scores", [])
        # We don't get raw masks from the API (only counts), so overlays are
        # boxes-only here. To see mask overlays, run via the experiments
        # runner which writes overlay PNGs to disk.
        overlay = draw_prediction(
            image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=None,
            score_threshold=thresholds["score_threshold"],
        )
        st.image(overlay, use_column_width=True, channels="RGB")
        st.caption(
            f"latency {pred['latency_ms']:.1f} ms · top-1 "
            f"{(pred.get('labels') or pred.get('topk_labels') or ['?'])[0]!r} "
            f"@ {(pred.get('scores') or pred.get('topk_scores') or [0.0])[0]:.3f}"
        )

# Metrics
st.subheader("Metrics summary")
metric_rows: list[dict] = []
for mname in models:
    preds = last["predictions"].get(mname, [])
    if not preds:
        continue
    latencies = [p["latency_ms"] for p in preds]
    detections = [p.get("n_masks", len(p.get("labels", []))) for p in preds]
    metric_rows.append(
        {
            "model": mname,
            "n_images": len(preds),
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "mean_detections": float(np.mean(detections)) if detections else 0.0,
        }
    )
if metric_rows:
    st.dataframe(metric_rows, use_container_width=True)

    # Bar chart of mean latency
    chart_data = {row["model"]: row["mean_latency_ms"] for row in metric_rows}
    st.bar_chart(chart_data, height=250)
else:
    st.info("No metrics to display yet.")

with st.expander("Raw prediction JSON"):
    render_json({mname: last["predictions"].get(mname, []) for mname in models})