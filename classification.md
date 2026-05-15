# Face Login + Emotion Recognition

Sistem ini menggabungkan **face recognition** (siapa kamu?) dengan **emotion/expression recognition** (kamu lagi ngerasa apa?) menggunakan webcam secara real-time.

---

## Alur Sistem

```
Webcam → Face Detection (Haar Cascade)
              ↓
    ┌─────────────────────┐
    │   Face Recognition  │  → Identifikasi identitas user (faces.pkl)
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │ Emotion Recognition │  → Klasifikasi ekspresi wajah
    └─────────────────────┘
```

---

## Klasifikasi Emosi

### Model yang Digunakan

| File | Kelas | Keterangan |
|------|-------|------------|
| `faceid/fer_trained.h5` | 7 kelas FER | Ditraining dari dataset Kaggle FER2013 via `train.py` |

### Kelas Emosi (7 kelas — FER2013 standard)

| Label | Emosi |
|-------|-------|
| 0 | Angry (Marah) |
| 1 | Disgust (Jijik) |
| 2 | Fear (Takut) |
| 3 | Happy (Senang) |
| 4 | Neutral (Netral) |
| 5 | Sad (Sedih) |
| 6 | Surprise (Kaget) |

### Arsitektur CNN (`train.py` / `fer_trained.h5`)

```
Input: 48×48 grayscale

Conv2D(32) → BN → MaxPool → Dropout(0.25)
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.25)
Flatten
Dense(256) → Dropout(0.5)
Dense(7, softmax)
```

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Input: gambar grayscale 48×48 dari ROI wajah

### Ekspresi Custom (`model.keras`)

Selain model FER di atas, ada model ekspresi **custom per-user** yang ditraining langsung dari webcam:
- 5 ekspresi yang bisa dikonfigurasi sendiri: `angry, happy, neutral, sad, surprised`
- Arsitektur: **MobileNetV2** (transfer learning, frozen) + head Dense kecil
- Input: 96×96 RGB

Model ini dipakai di fitur "Setup Ekspresi" dan diakses lewat tombol `E` di dashboard.

---

## Face Recognition

- Library: `face_recognition` (dlib-based)
- Database: `faces.pkl` — menyimpan face encodings (128-dim vector) per user
- Matching: Euclidean distance, threshold `< 0.5` dianggap match
- Registrasi: 5 foto wajah dari webcam, encoding disimpan ke pickle

---

## Struktur File

```
machinelearning/
├── faceid/
│   ├── app.py           # Aplikasi utama (webcam loop, register, login, ekspresi)
│   ├── train.py         # Training ulang model FER dari dataset Kaggle
│   ├── fer_trained.h5   # Model FER hasil training (7 kelas, 48×48 grayscale)
│   ├── model.keras      # Model ekspresi custom (MobileNetV2, 5 kelas, 96×96 RGB)
│   ├── faces.pkl        # Database face encodings per user
│   ├── labels.json      # Mapping indeks → nama kelas untuk model.keras
│   ├── requirements.txt # Dependensi Python
│   └── Makefile         # make run / make train / make install
└── archive/
    ├── train/           # Dataset FER2013 dari Kaggle (untuk train.py)
    └── test/            # Dataset test FER2013
```

---

## Cara Pakai

```bash
# Install dependencies
make install

# Jalankan aplikasi
make run

# Training ulang model FER (kalau mau update)
make train
```

### Kontrol di Aplikasi

| Tombol | Fungsi |
|--------|--------|
| `R` | Register wajah baru |
| `L` | Login dengan wajah |
| `E` | Cek ekspresi (dari dashboard) |
| `X` | Logout |
| `Q` | Keluar |
