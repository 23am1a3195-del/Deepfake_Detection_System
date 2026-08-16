import os
from PIL import Image

DATASET_PATH = os.path.join("dataset", "8020")

splits = ["train", "valid", "test"]
classes = ["fake", "real"]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print("=" * 60)
print("DEEPFAKE DATASET INSPECTION")
print("=" * 60)

total_images = 0
total_corrupted = 0

for split in splits:
    print(f"\n--- {split.upper()} ---")

    for class_name in classes:
        folder_path = os.path.join(DATASET_PATH, split, class_name)

        image_count = 0
        corrupted_count = 0
        image_sizes = []

        for filename in os.listdir(folder_path):

            if filename.lower().endswith(valid_extensions):

                image_path = os.path.join(folder_path, filename)

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    with Image.open(image_path) as img:
                        image_sizes.append(img.size)

                    image_count += 1

                except Exception:
                    corrupted_count += 1

        print(f"{class_name.upper():5} : {image_count} images")
        print(f"Corrupted: {corrupted_count}")

        if image_sizes:
            print(f"First image size: {image_sizes[0]}")

        total_images += image_count
        total_corrupted += corrupted_count

print("\n" + "=" * 60)
print(f"TOTAL VALID IMAGES : {total_images}")
print(f"TOTAL CORRUPTED    : {total_corrupted}")
print("=" * 60)
