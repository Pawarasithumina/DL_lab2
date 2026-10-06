import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


# Find original images
image_files = glob.glob(
    "raw/**/*.jpg",
    recursive=True
)

# Select one random image
image_path = random.choice(image_files)

image = Image.open(image_path).convert("RGB")


# Define the six augmentations separately
augmentations = [
    (
        "Horizontal Flip",
        transforms.RandomHorizontalFlip(p=1.0)
    ),

    (
        "Rotation",
        transforms.RandomRotation(15)
    ),

    (
        "Brightness",
        transforms.ColorJitter(
            brightness=0.5
        )
    ),

    (
        "Contrast",
        transforms.ColorJitter(
            contrast=0.5
        )
    ),

    (
        "Gaussian Blur",
        transforms.GaussianBlur(
            kernel_size=3
        )
    ),

    (
        "Perspective",
        transforms.RandomPerspective(
            distortion_scale=0.5,
            p=1.0
        )
    )
]


# Create figure
plt.figure(figsize=(12, 8))


# Show original image
plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")


# Apply and display each augmentation
for i, (name, transform) in enumerate(augmentations):

    augmented = transform(image)

    plt.subplot(2, 4, i + 2)
    plt.imshow(augmented)
    plt.title(name)
    plt.axis("off")


plt.tight_layout()


# Save visualization
os.makedirs(
    "visualizations",
    exist_ok=True
)

output_path = (
    "visualizations/augmentation_examples.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.show()

print("Saved:", output_path)
