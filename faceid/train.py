import os
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'archive', 'train')
MODEL_OUT    = os.path.join(os.path.dirname(__file__), 'fer_trained.h5')

IMG_SIZE = (48, 48)
BATCH    = 32
EPOCHS   = 20

print("Training pakai dataset Kaggle FER2013...")
print(f"Dataset: {os.path.abspath(DATASET_DIR)}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH, class_mode='categorical', subset='training'
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH, class_mode='categorical', subset='validation'
)

print(f"Kelas: {train_gen.class_indices}")
print(f"Train: {train_gen.samples} | Val: {val_gen.samples}")

model = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
]

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
          callbacks=callbacks, verbose=1)

print(f"\nModel tersimpan di: {MODEL_OUT}")
