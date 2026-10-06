import os
import glob
from PIL import Image
import torchvision.transforms as transforms


# Folder containing original images
raw_dir = "raw"

# Folder for augmented images
output_dir = "processed"

os.makedirs(output_dir, exist_ok=True)


# Augmentation pipeline
augmentation = transforms.Compose([
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
    )
])


# Find all images
image_files = glob.glob(
    os.path.join(raw_dir, "**", "*.jpg"),
    recursive=True
)


print("Total original images:", len(image_files))


for image_path in image_files:

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Get class folder
    relative_path = os.path.relpath(
        image_path,
        raw_dir
    )

    class_folder = os.path.dirname(relative_path)

    # Create corresponding output folder
    save_folder = os.path.join(
        output_dir,
        class_folder
    )

    os.makedirs(
        save_folder,
        exist_ok=True
    )

    # Get filename
    filename = os.path.basename(image_path)

    name = os.path.splitext(filename)[0]


    # Create 5 augmented images
    for i in range(5):

        augmented_image = augmentation(image)

        save_path = os.path.join(
            save_folder,
            f"{name}_aug{i + 1}.jpg"
        )

        augmented_image.save(save_path)


print("Augmentation completed.")
