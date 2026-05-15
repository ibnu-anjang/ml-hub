# FaceID

Aplikasi login berbasis wajah dengan deteksi ekspresi real-time menggunakan webcam.

## Fitur

- **Register** — simpan wajah baru ke database
- **Login** — verifikasi identitas lewat face recognition
- **Cek Ekspresi** — deteksi emosi setelah login (7 kelas: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

## Cara Pakai

```bash
# Install dependencies
make install

# Jalankan aplikasi
make run
```

### Kontrol

| Tombol | Fungsi |
|--------|--------|
| `R` | Register wajah baru |
| `L` | Login |
| `E` | Cek ekspresi (dari dashboard) |
| `X` | Logout |
| `Q` | Keluar |

## Training Ulang Model

Model FER ditraining dari dataset [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013). Letakkan dataset di `../archive/train/` lalu jalankan:

```bash
make train
```

## Stack

- Python 3.12
- TensorFlow / Keras
- OpenCV
- face_recognition (dlib)

## Struktur

```
faceid/
├── app.py           # Aplikasi utama
├── train.py         # Training model FER
├── expressions/     # Sample foto ekspresi custom
├── requirements.txt
└── Makefile
```
