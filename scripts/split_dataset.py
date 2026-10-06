import os
import glob
from sklearn.model_selection import train_test_split


# Class mapping
class_names = [
    "pawara",
    "uchitha",
    "hirusha",
    "unknown"
]

class_folders = {
    "pawara": "friend_1",
    "uchitha": "friend_2",
    "hirusha": "friend_3",
    "unknown": "unknown"
}


# Get all original image paths and labels
image_paths = []
labels = []


for label, class_name in enumerate(class_names):

    folder = class_folders[class_name]

    files = glob.glob(
        os.path.join("raw", folder, "*.jpg")
    )

    for file in files:
        image_paths.append(file)
        labels.append(label)


print("Total images:", len(image_paths))


# Stratified 80/20 split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))


# Create split folders
os.makedirs("processed/train", exist_ok=True)
os.makedirs("processed/val", exist_ok=True)


# Save training file list
with open("processed/train.txt", "w") as file:

    for path, label in zip(train_paths, train_labels):
        file.write(f"{path},{label}\n")


# Save validation file list
with open("processed/val.txt", "w") as file:

    for path, label in zip(val_paths, val_labels):
        file.write(f"{path},{label}\n")


print("Train and validation lists saved.")
