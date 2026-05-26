"""Split CCSN dataset → data/splits/{train,val,test}/{class}/ (symlink, no copy)."""
import os, random, shutil, sys
from pathlib import Path

import kagglehub

SEED = 42
RATIOS = (0.70, 0.15, 0.15)
BASE = Path(__file__).parent
SPLITS_DIR = BASE / "data" / "splits"

# 5-class taxonomy (merged version): kelas low-grey-overcast yang inherently
# mirip secara visual digabung biar model bisa belajar boundary yang jelas.
CLASS_MAP = {
    "Ci": "cirrus",
    "Cu": "cumulus",
    "Cb": "cumulonimbus",
    "Ac": "altocumulus",
    "Sc": "low_overcast",
    "St": "low_overcast",
    "Ns": "low_overcast",
}
DST_CLASSES = sorted(set(CLASS_MAP.values()))


def main() -> None:
    print("Resolving CCSN dataset path...")
    ds_root = Path(kagglehub.dataset_download(
        "mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database"
    )) / "CCSN_v2"
    if not ds_root.is_dir():
        sys.exit(f"Dataset folder not found: {ds_root}")

    if SPLITS_DIR.exists():
        print(f"Cleaning previous splits at {SPLITS_DIR}")
        shutil.rmtree(SPLITS_DIR)

    random.seed(SEED)
    counts = {split: {c: 0 for c in DST_CLASSES} for split in ("train", "val", "test")}

    for src_name, dst_name in CLASS_MAP.items():
        src = ds_root / src_name
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        random.shuffle(files)

        n = len(files)
        n_train = int(n * RATIOS[0])
        n_val = int(n * RATIOS[1])
        groups = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }
        for split, paths in groups.items():
            dst_dir = SPLITS_DIR / split / dst_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                # prefix dgn src_name biar nama unik (mencegah collision saat merge)
                link = dst_dir / f"{src_name}_{p.name}"
                os.symlink(p.resolve(), link)
            counts[split][dst_name] += len(paths)

    print(f"\n{'class':<16} {'train':>6} {'val':>6} {'test':>6} {'total':>6}")
    print("-" * 44)
    for c in DST_CLASSES:
        tr, va, te = counts["train"][c], counts["val"][c], counts["test"][c]
        print(f"{c:<16} {tr:>6} {va:>6} {te:>6} {tr+va+te:>6}")
    print("-" * 44)
    totals = [sum(counts[s].values()) for s in ("train", "val", "test")]
    print(f"{'TOTAL':<16} {totals[0]:>6} {totals[1]:>6} {totals[2]:>6} {sum(totals):>6}")
    print(f"\nSplits ready at: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
