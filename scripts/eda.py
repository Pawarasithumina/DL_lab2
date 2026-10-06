import torch
import matplotlib.pyplot as plt
import random


# -----------------------------
# Load datasets
# -----------------------------

train_data = torch.load(
    "processed/train.pt",
    weights_only=False
)

val_data = torch.load(
    "processed/val.pt",
    weights_only=False
)

train_images = train_data["images"]
train_labels = train_data["labels"]

val_images = val_data["images"]
val_labels = val_data["labels"]

class_names = train_data["class_names"]


# -----------------------------
# Print dataset information
# -----------------------------

print("Training dataset")
print("----------------")
print(f"Images shape: {train_images.shape}")
print(f"Labels shape: {train_labels.shape}")
print(f"Number of images: {len(train_images)}")

print()

print("Validation dataset")
print("------------------")
print(f"Images shape: {val_images.shape}")
print(f"Labels shape: {val_labels.shape}")
print(f"Number of images: {len(val_images)}")


# -----------------------------
# Class distributions
# -----------------------------

train_counts = []

val_counts = []

print()
print("Class distribution")
print("-------------------")

for i, class_name in enumerate(class_names):

    train_count = (train_labels == i).sum().item()
    val_count = (val_labels == i).sum().item()

    train_counts.append(train_count)
    val_counts.append(val_count)

    print(
        f"{class_name}: "
        f"train={train_count}, "
        f"validation={val_count}"
    )


# -----------------------------
# Train vs validation chart
# -----------------------------

x = range(len(class_names))

width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    [i - width / 2 for i in x],
    train_counts,
    width=width,
    label="Train"
)

plt.bar(
    [i + width / 2 for i in x],
    val_counts,
    width=width,
    label="Validation"
)

plt.xticks(
    list(x),
    class_names
)

plt.xlabel("Class")

plt.ylabel("Number of images")

plt.title("Training vs Validation Class Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    "visualizations/class_distribution.png",
    dpi=150
)

plt.show()


# -----------------------------
# Display 16 random training images
# -----------------------------

random_indices = random.sample(
    range(len(train_images)),
    16
)

plt.figure(figsize=(12, 10))

for i, index in enumerate(random_indices):

    image = train_images[index]

    label = train_labels[index].item()

    # Tensor: C,H,W
    # Matplotlib: H,W,C

    image = image.permute(1, 2, 0)

    plt.subplot(4, 4, i + 1)

    plt.imshow(image)

    plt.title(class_names[label])

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    "visualizations/random_16_train_images.png",
    dpi=150
)

plt.show()
