import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight 
import numpy as np
import kagglehub
import os

# --- 1. DOWNLOAD & SETUP ---
print("⬇️ Checking dataset...")
dataset_path = kagglehub.dataset_download("sachinkumar413/diabetic-retinopathy-dataset")

IMG_SIZE = 256
BATCH_SIZE = 32
MODEL_NAME = 'dr_model.h5'

# --- 2. DATA LOADERS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- 3. CALCULATE CLASS WEIGHTS (CRITICAL FIX) ---
# This calculates how "rare" each class is and assigns a higher penalty
# if the model gets the rare classes wrong.
print("⚖️ Calculating Class Weights...")
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"Class Weights: {class_weights_dict}")

# --- 4. BUILD MODEL ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) 
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 5. STAGE 1: WARM UP ---
print("🚀 STAGE 1: Warm Up...")
base_model.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    class_weight=class_weights_dict  # <--- APPLY WEIGHTS HERE
)

# --- 6. STAGE 2: FINE TUNING ---
print("🔓 STAGE 2: Fine Tuning...")
base_model.trainable = True
for layer in base_model.layers[:-50]: # Unfreeze more layers (50)
    layer.trainable = False

# Use Label Smoothing: Tells the model "Don't be 100% sure, be 90% sure".
# This prevents it from being confidently wrong on confusing images.
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss=loss_fn,  # <--- Use Smoothed Loss
              metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]

model.fit(
    train_generator,
    epochs=20, 
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict # <--- APPLY WEIGHTS HERE
)

# --- 7. SAVE ---
local_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(local_path, MODEL_NAME)
model.save(save_path)
print(f"✅ Model saved to {save_path}")