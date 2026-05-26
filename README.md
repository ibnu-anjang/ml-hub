# ML Hub

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-green.svg)](https://fastapi.tiangolo.com)

Kumpulan project ML demo dengan **satu web hub** terpadu.

## Project

| Project | Deskripsi | Status |
|---|---|---|
| **CloudLens** | Klasifikasi jenis awan (5-class: cirrus, cumulus, cumulonimbus, altocumulus, low_overcast) via MobileNetV2 transfer learning | ✅ trained (61.4% test acc) |
| **FaceID** | Pengenalan wajah + deteksi emosi (FER 7-class) via face_recognition + CNN | ✅ siap pakai |

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

## License

[MIT](LICENSE) © 2026 Ibnu Anjang Maulidi

Bebas dipakai, dimodifikasi, dan didistribusikan ulang. Cukup sertakan copyright + notice ini.
