"""CloudLens CLI: classify a sky photo.

Usage:
    python app.py path/to/photo.jpg
    python app.py path/to/photo.jpg --topk 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "model.keras"
LABELS_PATH = BASE / "labels.json"
IMG_SIZE = (224, 224)


def load_artifacts():
    if not MODEL_PATH.exists():
        sys.exit(
            f"❌ Model belum ada di {MODEL_PATH}\n"
            "   Jalankan dulu: make train"
        )
    if not LABELS_PATH.exists():
        sys.exit(f"❌ labels.json tidak ditemukan di {LABELS_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    labels = json.loads(LABELS_PATH.read_text())
    # {"0": {"key":..., "label":..., "desc":..., "weather":...}, ...}
    return model, labels


def predict_image(model, labels: dict, img_path: Path, topk: int = 1):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)[np.newaxis, ...]
    probs = model.predict(arr, verbose=0)[0]
    idx_sorted = np.argsort(probs)[::-1]
    results = []
    for i in idx_sorted[:topk]:
        info = labels[str(i)]
        results.append({
            "label":      info["label"],
            "confidence": float(probs[i]),
            "desc":       info["desc"],
            "weather":    info["weather"],
        })
    return results


def main():
    ap = argparse.ArgumentParser(description="CloudLens — klasifikasi jenis awan")
    ap.add_argument("image", type=Path, help="path ke foto langit")
    ap.add_argument("--topk", type=int, default=3, help="tampilkan N hasil teratas")
    args = ap.parse_args()

    if not args.image.exists():
        sys.exit(f"❌ File tidak ada: {args.image}")

    model, labels = load_artifacts()
    results = predict_image(model, labels, args.image, args.topk)

    print(f"\n📸 {args.image.name}")
    print("─" * 50)
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["confidence"] * 30)
        print(f"{i}. {r['label']:<14} {r['confidence']*100:5.1f}%  {bar}")
        if i == 1:
            print(f"   {r['desc']}")
            print(f"   ☁  {r['weather']}")
    print()


if __name__ == "__main__":
    main()
