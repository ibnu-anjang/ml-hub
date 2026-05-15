import cv2
import tensorflow as tf
import face_recognition
import numpy as np
import os
import time
import json
import pickle

BASE = os.path.dirname(__file__)
SAMPLES_DIR  = os.path.join(BASE, 'samples')
EXPR_DIR     = os.path.join(BASE, 'expressions')
MODEL_PATH   = os.path.join(BASE, 'model.keras')
LABELS_PATH  = os.path.join(BASE, 'labels.json')
FACES_DB     = os.path.join(BASE, 'faces.pkl')

EXPRESSIONS  = ['angry', 'happy', 'neutral', 'sad', 'surprised']
SAMPLES_EACH = 4
FER_MODEL_PATH = os.path.join(BASE, 'fer_trained.h5')
FER_LABELS    = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def detect_face_box(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    return faces if len(faces) > 0 else []

def draw_bar(frame, text, color=(200, 200, 200)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def get_text_input(prompt="Input:"):
    val = ""
    win = np.zeros((100, 420, 3), dtype=np.uint8)
    while True:
        d = win.copy()
        cv2.putText(d, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        cv2.putText(d, val + "|", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Input", d)
        k = cv2.waitKey(0)
        if k == 13 and val:
            cv2.destroyWindow("Input")
            return val
        elif k == 27:
            cv2.destroyWindow("Input")
            return None
        elif k == 8 and val:
            val = val[:-1]
        elif 32 <= k <= 126:
            val += chr(k)

def load_faces_db():
    if os.path.exists(FACES_DB):
        with open(FACES_DB, 'rb') as f:
            return pickle.load(f)
    return {}

def save_faces_db(db):
    with open(FACES_DB, 'wb') as f:
        pickle.dump(db, f)

# ─── Live camera loop (returns key pressed) ───────────────────────────────────

def live_loop(cap, hint, extra_hint="", check_face=True):
    """Tampilkan kamera + hint. Return key yang ditekan."""
    fc, last_boxes = 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        fc += 1

        if check_face and fc % 8 == 0:
            last_boxes = detect_face_box(frame)
        for (x, y, w, h) in last_boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 142, 247), 2)

        if extra_hint:
            cv2.putText(frame, extra_hint, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
        draw_bar(frame, hint)
        cv2.imshow("Face ID", frame)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            return k, frame

# ─── SCREEN: Register ─────────────────────────────────────────────────────────

def screen_register(cap):
    name = get_text_input("Nama (Enter=OK  Esc=Batal):")
    if not name:
        return

    save_dir = os.path.join(SAMPLES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    existing = len(os.listdir(save_dir))
    total, taken = 5, 0   # 5 foto cukup untuk face recognition

    while taken < total:
        start = time.time()
        while time.time() - start < 2:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            for (x, y, w, h) in detect_face_box(frame):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (79, 142, 247), 2)
            elapsed = time.time() - start
            bar_w = int((elapsed / 2) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-8),
                          (bar_w, frame.shape[0]), (79, 142, 247), -1)
            cv2.putText(frame, f"Register '{name}': foto {taken+1}/{total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Face ID", frame)
            cv2.waitKey(1)

        ret, snap = cap.read()
        if not ret: continue
        snap = cv2.flip(snap, 1)
        rgb  = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            db = load_faces_db()
            db.setdefault(name, []).append(encs[0])
            save_faces_db(db)
            fname = os.path.join(save_dir, f"{existing+taken+1}.jpg")
            cv2.imwrite(fname, snap)
            taken += 1

    # Feedback
    fb = np.zeros((80, 420, 3), dtype=np.uint8)
    cv2.putText(fb, f"'{name}' berhasil didaftarkan!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,80), 2)
    cv2.imshow("Face ID", fb)
    cv2.waitKey(1500)

# ─── SCREEN: Login ────────────────────────────────────────────────────────────

def screen_login(cap):
    db = load_faces_db()
    if not db:
        msg = np.zeros((80, 420, 3), dtype=np.uint8)
        cv2.putText(msg, "Belum ada wajah terdaftar!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,80,255), 2)
        cv2.imshow("Face ID", msg)
        cv2.waitKey(1500)
        return None

    # Countdown 3 detik
    start = time.time()
    snap  = None
    while time.time() - start < 3:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        remaining = 3 - int(time.time() - start)
        h, w = frame.shape[:2]
        cv2.putText(frame, str(remaining), (w//2-30, h//2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0,200,255), 8)
        cv2.putText(frame, "Hadapkan wajah ke kamera!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Face ID", frame)
        cv2.waitKey(1)
        snap = frame

    ret, snap = cap.read()
    snap = cv2.flip(snap, 1)
    rgb  = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)

    if not encs:
        msg = np.zeros((80, 420, 3), dtype=np.uint8)
        cv2.putText(msg, "Wajah tidak terdeteksi!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,80,255), 2)
        cv2.imshow("Face ID", msg)
        cv2.waitKey(1500)
        return None

    best_name, best_dist = 'unknown', 1.0
    for n, encs_list in db.items():
        dists = face_recognition.face_distance([np.array(e) for e in encs_list], encs[0])
        d = min(dists)
        if d < best_dist:
            best_dist, best_name = d, n

    class _Match:
        def __init__(self, label): self.label = label
    match = _Match(best_name if best_dist < 0.5 else 'unknown')

    if match.label == 'unknown':
        msg = np.zeros((80, 420, 3), dtype=np.uint8)
        cv2.putText(msg, "Akses ditolak!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 2)
        cv2.imshow("Face ID", msg)
        cv2.waitKey(1500)
        return None

    return match.label

# ─── SCREEN: Dashboard ────────────────────────────────────────────────────────

def screen_dashboard(cap, username):
    hint = f"Logged in: {username}  |  E=Cek Ekspresi  X=Logout"

    while True:
        k, frame = live_loop(cap, hint)
        if k == ord('x'):
            return
        elif k == ord('e'):
            screen_check_expression(cap)

# ─── SCREEN: Setup Ekspresi ───────────────────────────────────────────────────

def screen_setup_expressions(cap):
    # Kumpulin sampel
    for expr in EXPRESSIONS:
        save_dir = os.path.join(EXPR_DIR, expr)
        os.makedirs(save_dir, exist_ok=True)
        existing = len(os.listdir(save_dir))
        if existing >= SAMPLES_EACH:
            continue  # sudah cukup, skip

        taken = 0
        # Minta user siap
        info = np.zeros((120, 480, 3), dtype=np.uint8)
        cv2.putText(info, f"Ekspresi: {expr.upper()}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
        cv2.putText(info, "Siapkan ekspresi, lalu tekan SPASI", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
        cv2.imshow("Face ID", info)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            return  # batal

        while taken < SAMPLES_EACH:
            start = time.time()
            while time.time() - start < 1.5:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                for (x, y, w, h) in detect_face_box(frame):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (79,142,247), 2)
                elapsed = time.time() - start
                bar_w = int((elapsed / 1.5) * frame.shape[1])
                cv2.rectangle(frame, (0, frame.shape[0]-8),
                              (bar_w, frame.shape[0]), (0,200,255), -1)
                cv2.putText(frame, f"{expr.upper()}  {taken+1}/{SAMPLES_EACH}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow("Face ID", frame)
                cv2.waitKey(1)

            ret, snap = cap.read()
            if not ret: continue
            snap = cv2.flip(snap, 1)
            if len(detect_face_box(snap)) > 0:
                cv2.imwrite(os.path.join(save_dir, f"{existing+taken+1}.jpg"), snap)
                taken += 1

    # Training
    info = np.zeros((80, 420, 3), dtype=np.uint8)
    cv2.putText(info, "Training model... lihat terminal", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.imshow("Face ID", info)
    cv2.waitKey(1)
    _train_model()

    done = np.zeros((80, 420, 3), dtype=np.uint8)
    cv2.putText(done, "Model siap! Tekan E untuk cek ekspresi.", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,80), 2)
    cv2.imshow("Face ID", done)
    cv2.waitKey(2000)

def _train_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    IMG  = (96, 96)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, brightness_range=[0.7,1.3], horizontal_flip=True,
        zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
    )
    train_gen = datagen.flow_from_directory(EXPR_DIR, target_size=IMG,
                    batch_size=4, class_mode='categorical')

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG, 3), include_top=False, weights='imagenet'
    )
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(EXPRESSIONS), activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=15, verbose=1)
    model.save(MODEL_PATH)

    with open(LABELS_PATH, 'w') as f:
        json.dump(train_gen.class_indices, f)
    print("Model tersimpan.")

# ─── SCREEN: Cek Ekspresi ─────────────────────────────────────────────────────

def load_fer_model():
    return tf.keras.models.load_model(FER_MODEL_PATH)

def screen_check_expression(cap, _frame=None):
    model = load_fer_model()

    # Countdown 3 detik ambil frame
    start = time.time()
    snap  = None
    while time.time() - start < 3:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        remaining = 3 - int(time.time() - start)
        h, w = frame.shape[:2]
        cv2.putText(frame, str(remaining), (w//2-30, h//2+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0,200,255), 8)
        cv2.putText(frame, "Tunjukkan ekspresi kamu!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Face ID", frame)
        cv2.waitKey(1)
        snap = frame

    ret, snap = cap.read()
    snap = cv2.flip(snap, 1)

    gray  = cv2.cvtColor(snap, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    if len(faces) == 0:
        result = np.zeros((80, 420, 3), dtype=np.uint8)
        cv2.putText(result, "Wajah tidak terdeteksi!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,80,255), 2)
        cv2.imshow("Face ID", result)
        cv2.waitKey(1500)
        return

    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    inp = np.expand_dims(np.expand_dims(cv2.resize(roi, (48, 48)), -1), 0)
    cv2.normalize(inp, inp, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)

    pred = model.predict(inp, verbose=0)[0]
    idx  = int(np.argmax(pred))
    expr = FER_LABELS[idx]
    conf = int(pred[idx] * 100)

    # Tampilkan hasil di frame kamera
    cv2.rectangle(snap, (x, y), (x+w, y+h), (79, 142, 247), 2)
    cv2.putText(snap, f"{expr} ({conf}%)", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
    cv2.imshow("Face ID", snap)
    cv2.waitKey(3000)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak ditemukan!")
        return

    print("Face ID siap.")

    while True:
        k, _ = live_loop(cap,
                         hint="R=Register  L=Login  Q=Quit",
                         check_face=True)
        if k == ord('q'):
            break
        elif k == ord('r'):
            screen_register(cap)
        elif k == ord('l'):
            user = screen_login(cap)
            if user:
                screen_dashboard(cap, user)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
