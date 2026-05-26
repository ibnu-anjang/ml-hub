"""CloudLens training: MobileNetV2 transfer learning + fine-tuning.

Resource limits (set BEFORE importing TF so they actually take effect):
- 4 threads max across all BLAS / TF ops
- batch 16 to keep peak RAM ~2-3 GB
"""
import json
import os
from pathlib import Path

# IMPORTANT: must run before tensorflow / numpy import to take effect
_THREADS = "4"
for _v in (
    "TF_CPP_MIN_LOG_LEVEL",  # quieter TF logs
):
    os.environ.setdefault(_v, "2")
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
):
    os.environ.setdefault(_v, _THREADS)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# belt-and-suspenders: enforce TF thread limits at runtime too
tf.config.threading.set_inter_op_parallelism_threads(int(_THREADS))
tf.config.threading.set_intra_op_parallelism_threads(int(_THREADS))

BASE = Path(__file__).parent
SPLITS = BASE / "data" / "splits"
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.keras"
LABELS_PATH = BASE / "labels.json"

IMG_SIZE = (224, 224)
BATCH = 16          # was 32 → reduced to keep peak RAM low
SEED = 42

CLASS_INFO = {
    "altocumulus":  {"desc": "Gumpal sedang, ketinggian menengah",       "weather": "Cuaca tidak stabil"},
    "cirrus":       {"desc": "Tipis seperti bulu, ketinggian tinggi",    "weather": "Cuaca akan berubah dalam 24-48 jam"},
    "cumulonimbus": {"desc": "Sangat besar dan gelap, menjulang tinggi", "weather": "Badai petir dan hujan lebat"},
    "cumulus":      {"desc": "Gumpal putih tebal, dasar datar",          "weather": "Cuaca cerah dan baik"},
    "low_overcast": {"desc": "Lapisan awan rendah abu-abu (stratus/stratocumulus/nimbostratus)",
                     "weather": "Mendung — gerimis sampai hujan berkepanjangan"},
}


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS / "train", image_size=IMG_SIZE, batch_size=BATCH, seed=SEED, shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS / "val", image_size=IMG_SIZE, batch_size=BATCH, shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS / "test", image_size=IMG_SIZE, batch_size=BATCH, shuffle=False,
    )
    class_names = train_ds.class_names
    # small fixed prefetch buffer (no AUTOTUNE) → bounded RAM
    return (
        train_ds.prefetch(2),
        val_ds.prefetch(2),
        test_ds.prefetch(2),
        class_names,
    )


def build_model(n_classes: int):
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.20),
        layers.RandomContrast(0.20),
    ], name="augment")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # FROZEN — head-only training (v3 strategy)

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)            # extra dropout sebelum dense
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def compute_class_weights(train_ds, class_names):
    counts = np.zeros(len(class_names), dtype=np.int64)
    for _, y in train_ds.unbatch():
        counts[int(y.numpy())] += 1
    total = counts.sum()
    weights = total / (len(class_names) * counts)
    return {i: float(w) for i, w in enumerate(weights)}, counts.tolist()


def main():
    print(f"TF: {tf.__version__}")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    print("Classes:", class_names)

    class_weights, counts = compute_class_weights(train_ds, class_names)
    print("Train counts per class:", dict(zip(class_names, counts)))

    model, base = build_model(len(class_names))
    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor="val_loss", min_lr=1e-7),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss", mode="min"),
    ]

    # SINGLE PHASE: train head only (no fine-tuning to avoid overfit on small dataset).
    # Justified karena v1 (47%) > v2 fine-tuned (41%) → fine-tuning hurt pada 1116 sampel.
    print("\n=== Head-only training (MobileNetV2 frozen) ===")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=cb,
              class_weight=class_weights)

    # Evaluate
    print("\n=== Test set evaluation ===")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}")

    # Confusion matrix + report
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(p, axis=1).tolist())

    from sklearn.metrics import confusion_matrix, classification_report
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred)
    print("    " + " ".join(f"{c[:5]:>5}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i][:5]:>4} " + " ".join(f"{v:>5}" for v in row))
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    # Save labels.json (rich format)
    labels = {
        str(i): {
            "key":     name,
            "label":   name.capitalize(),
            "desc":    CLASS_INFO[name]["desc"],
            "weather": CLASS_INFO[name]["weather"],
        }
        for i, name in enumerate(class_names)
    }
    LABELS_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"\nSaved labels → {LABELS_PATH}")
    print(f"Best model checkpoint → {MODEL_PATH}")


if __name__ == "__main__":
    main()
