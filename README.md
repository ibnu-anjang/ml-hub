# ML Hub

Kumpulan project ML demo dengan **satu web hub** terpadu.

## Project

| Project | Deskripsi | Status |
|---|---|---|
| **CloudLens** | Klasifikasi 7 jenis awan dari foto langit (MobileNetV2) | ⏸ model belum trained |
| **FaceID** | Pengenalan wajah + deteksi emosi (FER 7-class) | ✅ siap pakai |

## Cara Pakai

### Web Hub
```bash
make web
# → http://localhost:8000
```

Landing page menampilkan kartu per project. Klik untuk masuk ke UI tiap module.

### CLI per project

```bash
# CloudLens
make cloudlens-train                              # training (~30-40 menit, sudah throttled)
make cloudlens-app ARGS="path/ke/foto.jpg"        # inference single file

# FaceID (original webcam app)
cd faceid && make run
```

## Struktur

```
machinelearning/
├── Makefile             # entry-point hub
├── faceid/              # face recognition + emotion
│   ├── app.py           # webcam app (standalone)
│   ├── faces.pkl        # database wajah
│   ├── model.keras      # personal expressions
│   ├── fer_trained.h5   # FER 7-class
│   └── venv312/         # shared venv untuk semua project
├── cloudlens/           # klasifikasi awan
│   ├── train.py         # training script
│   ├── app.py           # CLI inference
│   ├── prepare_data.py  # dataset split
│   ├── catatan.md       # log progres build cloudlens
│   ├── data/splits/     # train/val/test (symlink ke CCSN cache)
│   └── models/          # output: model.keras
└── web/                 # HUB FastAPI
    ├── main.py
    ├── modules/
    │   ├── cloudlens.py
    │   └── faceid.py
    ├── templates/
    └── static/
```

## API Endpoints (hub web)

### CloudLens
- `GET /api/cloudlens/status` — cek apakah model siap
- `POST /api/cloudlens/predict` — multipart `file=<image>`, return top-K + confidence + deskripsi cuaca

### FaceID
- `GET /api/faceid/status` — daftar user terdaftar
- `POST /api/faceid/recognize` — match wajah ke database
- `POST /api/faceid/emotion` — deteksi 7 emosi
- `POST /api/faceid/analyze` — recognize + emotion dalam 1 request

Input bisa **upload file** (`file=<image>`) atau **base64 data URL** dari webcam (`image=data:image/jpeg;base64,...`).

## Nambah Project Baru

1. Bikin folder `machinelearning/projectX/`
2. Tulis `web/modules/projectX.py` dengan `router = APIRouter(prefix="/api/projectX")`
3. Mount di `web/main.py`: `app.include_router(projectX.router)`
4. Tambah template `web/templates/projectX.html` + entry di `PROJECTS` list
5. Done — kartu landing otomatis muncul

## Stack

- Python 3.12 (via pyenv)
- TensorFlow 2.21 + Keras 3.14
- FastAPI + Uvicorn
- face_recognition (dlib) untuk pengenalan wajah
- Vanilla HTML/JS frontend (no build step)
