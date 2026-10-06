import os
import torch
from PIL import Image
from torchvision import transforms


class_names = [
    "pawara",
    "uchitha",
    "hirusha",
    "unknown"
]


# Resize every image to the same size
image_transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor()
])


def load_split(file_path):

    images = []
    labels = []
    paths = []

    with open(file_path, "r") as file:

        for line in file:

            path, label = line.strip().split(",")

            image = Image.open(path).convert("RGB")

            image = image_transform(image)

            images.append(image)
            labels.append(int(label))
            paths.append(path)

    images = torch.stack(images)
    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return images, labels, paths


# Load training data
train_images, train_labels, train_paths = load_split(
    "processed/train.txt"
)


# Load validation data
val_images, val_labels, val_paths = load_split(
    "processed/val.txt"
)


# Create training dataset
train_data = {
    "images": train_images,
    "labels": train_labels,
    "paths": train_paths,
    "class_names": class_names
}


# Create validation dataset
val_data = {
    "images": val_images,
    "labels": val_labels,
    "paths": val_paths,
    "class_names": class_names
}


# Save datasets
torch.save(
    train_data,
    "processed/train.pt"
)

torch.save(
    val_data,
    "processed/val.pt"
)


# Print information
print()
print("Training dataset")
print("----------------")
print("Images shape:", train_images.shape)
print("Labels shape:", train_labels.shape)
print("Number of images:", len(train_images))
print(
    "File size:",
    round(
        os.path.getsize("processed/train.pt") / 1024,
        2
    ),
    "KB"
)


print()
print("Validation dataset")
print("------------------")
print("Images shape:", val_images.shape)
print("Labels shape:", val_labels.shape)
print("Number of images:", len(val_images))
print(
    "File size:",
    round(
        os.path.getsize("processed/val.pt") / 1024,
        2
    ),
    "KB"
)


print()
print("Dataset files created.")
