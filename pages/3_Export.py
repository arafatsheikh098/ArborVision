"""Page 3 — Export experiment results and reproducibility config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from arborvision.config import RESULTS_DIR
from arborvision.ui.client import get, render_json


st.set_page_config(page_title="Export — ArborVision", page_icon="💾", layout="wide")
st.title("💾 Export")

st.markdown(
    """
    Download a **reproducibility config** (model + weights + device +
    thresholds + package versions), the **metrics CSV**, and the **full
    results JSON** for any experiment that was saved to disk by the
    experiment runner.
    """
)

# List past experiments
exps = get("/experiments").get("experiments", [])

if exps:
    st.subheader("Past experiments (from FastAPI)")
    df = pd.DataFrame(exps)
    st.dataframe(df, use_container_width=True)
    selected_id = st.selectbox(
        "Select an experiment to download",
        options=[e["experiment_id"] for e in exps],
    )
    if selected_id:
        exp_path = RESULTS_DIR / selected_id
        config_path = exp_path / "config.json"
        results_path = exp_path / "results.json"
        metrics_path = exp_path / "metrics.csv"
        zip_path = exp_path / "experiment.zip"

        if config_path.exists():
            st.download_button(
                "Download config.json",
                data=config_path.read_bytes(),
                file_name=f"{selected_id}_config.json",
                mime="application/json",
            )
        if results_path.exists():
            st.download_button(
                "Download results.json",
                data=results_path.read_bytes(),
                file_name=f"{selected_id}_results.json",
                mime="application/json",
            )
        if metrics_path.exists():
            st.download_button(
                "Download metrics.csv",
                data=metrics_path.read_bytes(),
                file_name=f"{selected_id}_metrics.csv",
                mime="text/csv",
            )
        if zip_path.exists():
            st.download_button(
                "Download experiment.zip (everything)",
                data=zip_path.read_bytes(),
                file_name=f"{selected_id}.zip",
                mime="application/zip",
            )
        if st.checkbox("Show full results JSON inline"):
            render_json(json.loads(results_path.read_text()))
else:
    st.info("No past experiments found on disk.")

# Also allow downloading the in-session last run
st.markdown("---")
st.subheader("In-session last run")
last = st.session_state.get("last_predictions")
if last:
    st.download_button(
        "Download last run (JSON)",
        data=json.dumps(last, indent=2, default=str).encode("utf-8"),
        file_name="arborvision_last_run.json",
        mime="application/json",
    )
    # CSV
    rows = []
    for mname, preds in last.get("predictions", {}).items():
        for p in preds:
            labels = p.get("labels") or p.get("topk_labels") or [""]
            scores = p.get("scores") or p.get("topk_scores") or [0.0]
            rows.append(
                {
                    "model": mname,
                    "top1_label": labels[0] if labels else "",
                    "top1_score": float(scores[0]) if scores else 0.0,
                    "latency_ms": p.get("latency_ms", 0.0),
                    "n_detections": p.get("n_masks", len(p.get("labels", []))),
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        st.download_button(
            "Download last run (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="arborvision_last_run.csv",
            mime="text/csv",
        )
else:
    st.info("No in-session results to export yet.")