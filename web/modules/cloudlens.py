"""CloudLens module — POST /api/cloudlens/predict (multipart image upload)."""
import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "cloudlens"
MODEL_PATH = ROOT / "models" / "model.keras"
LABELS_PATH = ROOT / "labels.json"
IMG_SIZE = (224, 224)

router = APIRouter(prefix="/api/cloudlens", tags=["cloudlens"])

# Lazy state — only load model on first request after it appears on disk
_state = {"model": None, "labels": None, "mtime": 0.0}


def _try_load():
    """Reload model if file exists and changed since last load."""
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return False
    mtime = MODEL_PATH.stat().st_mtime
    if _state["model"] is not None and mtime == _state["mtime"]:
        return True
    import tensorflow as tf  # local import to avoid TF startup if module unused
    _state["model"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _state["labels"] = json.loads(LABELS_PATH.read_text())
    _state["mtime"] = mtime
    return True


@router.get("/status")
def status():
    ready = _try_load()
    return {
        "ready": ready,
        "model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "hint": None if ready else "Model belum dilatih. Jalankan: cd cloudlens && make train",
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    if not _try_load():
        raise HTTPException(
            status_code=503,
            detail="Model belum dilatih. Jalankan `make train` di folder cloudlens.",
        )

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize(IMG_SIZE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal baca gambar: {e}")

    arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
    t0 = time.perf_counter()
    probs = _state["model"].predict(arr, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    idx_sorted = np.argsort(probs)[::-1][:topk]
    labels = _state["labels"]
    results = [{
        "label":      labels[str(int(i))]["label"],
        "key":        labels[str(int(i))]["key"],
        "confidence": float(probs[i]),
        "desc":       labels[str(int(i))]["desc"],
        "weather":    labels[str(int(i))]["weather"],
    } for i in idx_sorted]

    return {"results": results, "latency_ms": round(latency_ms, 1)}
