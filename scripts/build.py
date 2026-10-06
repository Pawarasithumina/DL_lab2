import os
import json
import random

import torch
from PIL import Image
from torchvision import transforms


# -----------------------------
# Settings
# -----------------------------

DATASET_DIR = "./fmfdataset"

IMAGE_SIZE = 224
AUGMENT_FACTOR = 6

TRAIN_SPLIT = "processed/train.txt"
VAL_SPLIT = "processed/val.txt"

TRAIN_OUTPUT = "processed/train.pt"
VAL_OUTPUT = "processed/val.pt"


# -----------------------------
# Load class names
# -----------------------------

with open("config.json", "r") as f:
    config = json.load(f)

class_names = config["class_names"]

class_to_idx = {
    name: i for i, name in enumerate(class_names)
}


# -----------------------------
# Folder to class mapping
# -----------------------------

folder_to_class = {
    "friend_1": "pawara",
    "friend_2": "uchitha",
    "friend_3": "hirusha",
    "unknown": "unknown"
}


# -----------------------------
# Training augmentations
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.GaussianBlur(
        kernel_size=3
    ),

    transforms.RandomPerspective(
        distortion_scale=0.2,
        p=0.5
    ),

    transforms.ToTensor()
])


# -----------------------------
# Validation transform
# -----------------------------

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# -----------------------------
# Read split file
# -----------------------------

def read_split(filename):

    paths = []
    labels = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # train.txt format:
            # path,label

            if "," in line:

                path, label = line.rsplit(",", 1)

                path = path.strip()
                label = int(label.strip())

            # Also support:
            # path label

            elif " " in line:

                parts = line.split()

                path = parts[0]
                label = int(parts[1])

            # If only path exists
            else:

                path = line

                folder = path.split("/")[1]

                class_name = folder_to_class[folder]

                label = class_to_idx[class_name]

            paths.append(path)
            labels.append(label)

    return paths, labels


# -----------------------------
# Build training dataset
# -----------------------------

def build_training_dataset():

    paths, labels = read_split(TRAIN_SPLIT)

    print("Building training dataset...")
    print(f"Original training images: {len(paths)}")
    print(f"Augmentation factor: {AUGMENT_FACTOR}")

    images = []
    final_labels = []
    final_paths = []

    for path, label in zip(paths, labels):

        image = Image.open(path).convert("RGB")

        for _ in range(AUGMENT_FACTOR):

            augmented_image = train_transform(image)

            images.append(augmented_image)
            final_labels.append(label)
            final_paths.append(path)

    images = torch.stack(images)
    labels_tensor = torch.tensor(
        final_labels,
        dtype=torch.long
    )

    dataset = {
        "images": images,
        "labels": labels_tensor,
        "paths": final_paths,
        "class_names": class_names
    }

    torch.save(dataset, TRAIN_OUTPUT)

    print()
    print("Training dataset created.")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Number of images: {len(images)}")

    print()
    print("Training samples per class:")

    for class_name in class_names:

        class_index = class_to_idx[class_name]

        count = (labels_tensor == class_index).sum().item()

        print(f"{class_name} : {count}")


# -----------------------------
# Build validation dataset
# -----------------------------

def build_validation_dataset():

    paths, labels = read_split(VAL_SPLIT)

    print()
    print("Building validation dataset...")
    print(f"Validation images: {len(paths)}")

    images = []
    final_labels = []
    final_paths = []

    for path, label in zip(paths, labels):

        image = Image.open(path).convert("RGB")

        image_tensor = val_transform(image)

        images.append(image_tensor)
        final_labels.append(label)
        final_paths.append(path)

    images = torch.stack(images)

    labels_tensor = torch.tensor(
        final_labels,
        dtype=torch.long
    )

    dataset = {
        "images": images,
        "labels": labels_tensor,
        "paths": final_paths,
        "class_names": class_names
    }

    torch.save(dataset, VAL_OUTPUT)

    print()
    print("Validation dataset created.")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Number of images: {len(images)}")

    print()
    print("Validation samples per class:")

    for class_name in class_names:

        class_index = class_to_idx[class_name]

        count = (labels_tensor == class_index).sum().item()

        print(f"{class_name} : {count}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    build_training_dataset()

    build_validation_dataset()

    print()
    print("Dataset building completed.")
