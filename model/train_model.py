import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# SETTINGS
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "dataset/8020"

# -----------------------------
# LOAD DATASET
# -----------------------------
train_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/valid",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/test",
    labels="inferred",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:", train_data.class_names)

# -----------------------------
# PERFORMANCE
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
valid_data = valid_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -----------------------------
# MOBILENETV2
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# DISPLAY MODEL
# -----------------------------
model.summary()

print("\nMobileNetV2 model created successfully!")
# -----------------------------
# TRAINING CALLBACKS
# -----------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_deepfake_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

print("\nStarting model training...")

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)

print("\nTraining completed successfully!")

# Save final model
model.save("model/deepfake_model_final.keras")

print("Final model saved successfully!")
