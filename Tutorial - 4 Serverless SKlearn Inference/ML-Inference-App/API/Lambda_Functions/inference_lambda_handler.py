import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("inference")

# ---------- Config ----------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model-inference-1.0.joblib"   # same folder as this file
MODEL = None  # cached

CLASS_NAMES_ENV = os.getenv("CLASS_NAMES")
DEFAULT_CLASS_NAMES = (
    [c.strip() for c in CLASS_NAMES_ENV.split(",")] if CLASS_NAMES_ENV else None
)

# ---------- Helpers ----------
def _load_model() -> Any:
    global MODEL
    if MODEL is not None:
        return MODEL
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Place the .joblib file in the same directory as lambda_function.py."
        )
    logger.info("Loading model from %s", MODEL_PATH)
    MODEL = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    return MODEL


def _parse_event(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body", event)
    if isinstance(body, str):
        body = json.loads(body or "{}")
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    return body


def _predict(instances: List[List[float]]) -> Tuple[List[str], List[float]]:
    model = _load_model()
    X = np.asarray(instances, dtype=float)

    preds = model.predict(X)

    # Determine class names (for human-readable labels if available)
    class_names = getattr(model, "classes_", None)
    if class_names is None and hasattr(model, "named_steps"):
        try:
            last_step = list(model.named_steps.keys())[-1]
            class_names = getattr(model.named_steps[last_step], "classes_", None)
        except Exception:
            class_names = None
    if class_names is None and DEFAULT_CLASS_NAMES:
        class_names = np.array(DEFAULT_CLASS_NAMES)

    if class_names is not None and len(class_names) > int(np.max(preds)):
        pred_labels = [str(class_names[i]) for i in preds]
    else:
        pred_labels = [str(p) for p in preds]

    # Max probability per instance (single float), if supported
    max_probs: List[float] = []
    try:
        proba = model.predict_proba(X)          # shape: [n_samples, n_classes]
        max_probs = np.max(proba, axis=1).tolist()
    except Exception:
        # Model doesn't support predict_proba; return empty list
        max_probs = []

    return pred_labels, max_probs


def _response(status_code: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        },
        "body": json.dumps(payload),
        "isBase64Encoded": False,
    }

# ---------- Lambda entrypoint ----------
def lambda_handler(event, context):
    start_perf = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Event received (keys): %s", list(event.keys()) if isinstance(event, dict) else type(event))

    if isinstance(event, dict) and event.get("httpMethod") == "OPTIONS":
        return _response(200, {"Message": "OK"})

    try:
        body = _parse_event(event)
        instances = body.get("instances")

        if not instances or not isinstance(instances, list):
            return _response(400, {"Error": "JSON must include 'instances': [[...], ...]."})

        pred_labels, max_probs = _predict(instances)

        tat_ms = int((time.perf_counter() - start_perf) * 1000)
        result = {
            "Predictions": pred_labels,
            "Probabilities": max_probs,   # now a single max-prob per instance
            "Timestamp": timestamp,
            "TatMs": tat_ms
        }

        logger.info("Result: %s", result)
        return _response(200, result)

    except Exception as e:
        tat_ms = int((time.perf_counter() - start_perf) * 1000)
        logger.exception("Inference error (TatMs=%d)", tat_ms)
        return _response(500, {"Error": str(e), "Timestamp": timestamp, "TatMs": tat_ms})
