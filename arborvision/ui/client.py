"""Small HTTP client used by the Streamlit UI to talk to the FastAPI backend.

Centralized here so all pages share a single point of configuration (the
API base URL stored in ``st.session_state``) and consistent error handling.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def api_url() -> str:
    return st.session_state.get("api_url", "http://127.0.0.1:8765").rstrip("/")


def get(path: str) -> dict[str, Any]:
    """GET a JSON resource from the API; surface errors to the Streamlit UI."""
    import httpx

    url = f"{api_url()}{path}"
    try:
        r = httpx.get(url, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        st.error(f"API GET {url} failed: {e}")
        return {}


def post_predict(files: list, model_names: list[str]) -> dict[str, Any]:
    """POST /predict with one or more uploaded images."""
    import httpx

    url = f"{api_url()}/predict"
    try:
        # httpx 0.28 expects (field_name, file_value) tuples where
        # file_value is a 2- or 3-tuple of (filename, fileobj[, content_type]).
        # The dict form is NOT supported in this version — that's the bug
        # that caused the original "too many values to unpack" error.
        file_payload = []
        for f in files:
            file_payload.append(
                (
                    "files",  # form field name; matches FastAPI's parameter
                    (
                        f.name,
                        f.getvalue(),
                        f.type or "application/octet-stream",
                    ),
                )
            )
        r = httpx.post(
            url,
            files=file_payload,
            data={"model_names": ",".join(model_names)},
            timeout=300.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        st.error(f"API POST /predict failed: {e}")
        return {}


def fallback_local_predict(image_bytes: bytes, model_names: list[str]) -> dict[str, Any]:
    """Run models in-process if the API is unreachable.

    Useful for the MVP where users may not have started the API. This
    calls the adapter registry directly; results are returned in the same
    shape as the API's ``MultiPredictResponse``.
    """
    import numpy as np

    from arborvision.data.loader import decode_upload
    from arborvision.models import get_model

    try:
        image = decode_upload(image_bytes)
    except Exception as e:
        st.error(f"Could not decode image: {e}")
        return {"predictions": {}, "total_latency_ms": 0.0}

    predictions: dict[str, list[dict]] = {n: [] for n in model_names}
    total_latency = 0.0
    for n in model_names:
        m = get_model(n)
        m.warmup()
        pred = m.timed_predict(image)
        total_latency += pred.latency_ms
        predictions[n].append(
            {
                "model": n,
                "latency_ms": pred.latency_ms,
                "labels": pred.labels,
                "scores": pred.scores,
                "boxes": [list(b) for b in pred.boxes],
                "n_masks": len(pred.masks),
                "topk_labels": pred.topk_labels,
                "topk_scores": pred.topk_scores,
            }
        )
    return {"predictions": predictions, "total_latency_ms": total_latency}


def is_api_alive() -> bool:
    """Cheap health check used to decide between API and local fallback."""
    import httpx

    try:
        r = httpx.get(f"{api_url()}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def render_json(obj: Any) -> None:
    """Render a Python object as a pretty code block."""
    st.code(json.dumps(obj, indent=2, default=str), language="json")