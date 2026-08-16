import tensorflow as tf

# Dataset location
DATASET_PATH = "dataset/8020"

# Image settings
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/train",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load validation data
valid_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/valid",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load testing data
test_data = tf.keras.utils.image_dataset_from_directory(
    f"{DATASET_PATH}/test",
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize pixel values from 0-255 to 0-1
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_data = train_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

valid_data = valid_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_data = test_data.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Improve loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
valid_data = valid_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

print("Dataset loading completed successfully!")
print("Classes:", train_data.class_names if hasattr(train_data, "class_names") else "fake / real")