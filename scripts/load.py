import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -----------------------------
# ImageNet normalization
# -----------------------------

mean = [
    0.485,
    0.456,
    0.406
]

std = [
    0.229,
    0.224,
    0.225
]


# -----------------------------
# Dataset class
# -----------------------------

class FindMyFriendDataset(Dataset):

    def __init__(
        self,
        pt_file,
        transform=None
    ):

        data = torch.load(
            pt_file,
            weights_only=False
        )

        self.images = data["images"]
        self.labels = data["labels"]
        self.paths = data["paths"]
        self.class_names = data["class_names"]

        self.transform = transform


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# -----------------------------
# Transforms
# -----------------------------

train_transform = transforms.Normalize(
    mean=mean,
    std=std
)

val_transform = transforms.Normalize(
    mean=mean,
    std=std
)


# -----------------------------
# Create DataLoaders
# -----------------------------

def create_dataloaders(
    batch_size=8
):

    train_dataset = FindMyFriendDataset(
        "processed/train.pt",
        transform=train_transform
    )

    val_dataset = FindMyFriendDataset(
        "processed/val.pt",
        transform=val_transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        train_dataset.class_names
    )


# -----------------------------
# Test DataLoader
# -----------------------------

train_loader, val_loader, class_names = create_dataloaders(
    batch_size=8
)


images, labels = next(
    iter(train_loader)
)


print("Training dataset:", len(train_loader.dataset))
print("Validation dataset:", len(val_loader.dataset))

print()

print("Batch shape:", images.shape)

print()

print("Labels:", labels)

print()

print(
    "Label names:",
    [
        class_names[label.item()]
        for label in labels
    ]
)
