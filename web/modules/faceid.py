"""FaceID module — face recognition + emotion classification.

Endpoints:
- POST /api/faceid/recognize  → match face against faces.pkl
- POST /api/faceid/emotion    → predict emotion from face crop (FER 7-class)
- POST /api/faceid/analyze    → both, single request

Input: multipart image OR base64 data URL (from webcam snapshot).
"""
import base64
import io
import pickle
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "faceid"
FACES_DB_PATH = ROOT / "faces.pkl"
FER_MODEL_PATH = ROOT / "fer_trained.h5"
SAMPLES_DIR = ROOT / "samples"
REQUIRED_SHOTS = 5  # jumlah foto per user untuk akurasi recognition

FER_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

router = APIRouter(prefix="/api/faceid", tags=["faceid"])

_state = {"faces_db": None, "fer_model": None}


def _load_models():
    if _state["fer_model"] is None:
        import tensorflow as tf
        _state["fer_model"] = tf.keras.models.load_model(FER_MODEL_PATH, compile=False)
    if _state["faces_db"] is None and FACES_DB_PATH.exists():
        with open(FACES_DB_PATH, "rb") as f:
            _state["faces_db"] = pickle.load(f)


def _decode_image(file: UploadFile | None, data_url: str | None) -> np.ndarray:
    """Return RGB np.uint8 array (H, W, 3)."""
    if file is not None:
        raw = file.file.read()
    elif data_url:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
    else:
        raise HTTPException(400, "Sertakan `file` (upload) atau `image` (data URL).")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Gambar tidak valid: {e}")
    return np.array(img)


def _recognize(rgb: np.ndarray, tolerance: float = 0.5) -> dict:
    """Find best face match in faces.pkl."""
    import face_recognition
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        return {"matched": False, "reason": "no_face_detected"}
    encs = face_recognition.face_encodings(rgb, boxes)
    if not encs:
        return {"matched": False, "reason": "no_encoding"}

    db = _state["faces_db"] or {}
    if not db:
        return {"matched": False, "reason": "empty_db",
                "hint": "Belum ada wajah terdaftar di faces.pkl"}

    target = encs[0]
    best_name, best_dist = None, 1e9
    for name, enc_list in db.items():
        for ref in enc_list:
            d = float(np.linalg.norm(np.array(ref) - target))
            if d < best_dist:
                best_dist, best_name = d, name

    matched = best_dist < tolerance
    top, right, bottom, left = boxes[0]
    return {
        "matched":   matched,
        "name":      best_name if matched else None,
        "distance":  round(best_dist, 4),
        "tolerance": tolerance,
        "box":       {"top": top, "right": right, "bottom": bottom, "left": left},
    }


def _emotion(rgb: np.ndarray, box: dict | None = None) -> dict:
    """Predict emotion. If box given, crop to face first."""
    img = Image.fromarray(rgb)
    if box:
        img = img.crop((box["left"], box["top"], box["right"], box["bottom"]))
    img_gray = img.convert("L").resize((48, 48))
    arr = np.asarray(img_gray, dtype=np.float32)[np.newaxis, ..., np.newaxis] / 255.0
    probs = _state["fer_model"].predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "emotion":    FER_LABELS[idx],
        "confidence": float(probs[idx]),
        "scores":     {FER_LABELS[i]: float(probs[i]) for i in range(len(FER_LABELS))},
    }


@router.get("/status")
def status():
    info = {
        "fer_model_exists": FER_MODEL_PATH.exists(),
        "faces_db_exists":  FACES_DB_PATH.exists(),
        "registered_users": [],
    }
    if FACES_DB_PATH.exists():
        with open(FACES_DB_PATH, "rb") as f:
            db = pickle.load(f)
        info["registered_users"] = list(db.keys())
    return info


@router.post("/recognize")
async def recognize(file: UploadFile = File(None), image: str = Form(None)):
    _load_models()
    rgb = _decode_image(file, image)
    return _recognize(rgb)


@router.post("/emotion")
async def emotion(file: UploadFile = File(None), image: str = Form(None)):
    _load_models()
    rgb = _decode_image(file, image)
    return _emotion(rgb)


@router.post("/analyze")
async def analyze(file: UploadFile = File(None), image: str = Form(None)):
    """Recognize + emotion in one call."""
    _load_models()
    rgb = _decode_image(file, image)
    rec = _recognize(rgb)
    emo = _emotion(rgb, rec.get("box"))
    return {"recognition": rec, "emotion": emo}


@router.post("/register")
async def register(
    name: str = Form(...),
    file: UploadFile = File(None),
    image: str = Form(None),
):
    """Append one face encoding for `name` to faces.pkl. Frontend kirim 5x untuk dapat 5 encoding."""
    import face_recognition
    name = name.strip()
    if not name:
        raise HTTPException(400, "Nama tidak boleh kosong.")
    if not name.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(400, "Nama hanya boleh huruf/angka/underscore/dash.")

    _load_models()
    rgb = _decode_image(file, image)
    boxes = face_recognition.face_locations(rgb)
    if not boxes:
        raise HTTPException(422, "Tidak ada wajah terdeteksi di frame.")
    encs = face_recognition.face_encodings(rgb, boxes)
    if not encs:
        raise HTTPException(422, "Wajah tidak bisa di-encode, coba lighting lebih baik.")

    # Load (don't rely on cached state — registration mutates)
    db = {}
    if FACES_DB_PATH.exists():
        with open(FACES_DB_PATH, "rb") as f:
            db = pickle.load(f)
    db.setdefault(name, []).append(encs[0])
    with open(FACES_DB_PATH, "wb") as f:
        pickle.dump(db, f)
    _state["faces_db"] = db  # update in-memory cache

    # Save reference photo (optional, untuk debugging visual)
    sample_dir = SAMPLES_DIR / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in sample_dir.glob("*.jpg"))
    Image.fromarray(rgb).save(sample_dir / f"{existing + 1}.jpg", "JPEG", quality=85)

    return {
        "name":          name,
        "encodings":     len(db[name]),
        "required":      REQUIRED_SHOTS,
        "complete":      len(db[name]) >= REQUIRED_SHOTS,
    }


@router.delete("/register/{name}")
def unregister(name: str):
    """Hapus user dari faces.pkl + foto sample-nya."""
    import shutil
    if not FACES_DB_PATH.exists():
        raise HTTPException(404, "Database belum ada.")
    with open(FACES_DB_PATH, "rb") as f:
        db = pickle.load(f)
    if name not in db:
        raise HTTPException(404, f"User '{name}' tidak ditemukan.")
    del db[name]
    with open(FACES_DB_PATH, "wb") as f:
        pickle.dump(db, f)
    _state["faces_db"] = db

    sample_dir = SAMPLES_DIR / name
    if sample_dir.is_dir():
        shutil.rmtree(sample_dir)

    return {"deleted": name, "remaining": list(db.keys())}
