import time

import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small


# -----------------------------
# Load training dataset
# -----------------------------

data = torch.load(
    "processed/train.pt",
    weights_only=False
)


images = data["images"]
labels = data["labels"]

class_names = data["class_names"]


# -----------------------------
# Dataset
# -----------------------------

class SimpleDataset(Dataset):

    def __init__(
        self,
        images,
        labels
    ):

        self.images = images
        self.labels = labels


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        return (
            self.images[index],
            self.labels[index]
        )


dataset = SimpleDataset(
    images,
    labels
)


# -----------------------------
# Device
# -----------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# -----------------------------
# Compare batch sizes
# -----------------------------

for batch_size in [8, 64]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


    model = mobilenet_v3_small(
        weights=None,
        num_classes=len(class_names)
    )


    model = model.to(device)

    model.eval()


    start = time.time()


    with torch.no_grad():

        for batch_images, batch_labels in loader:

            batch_images = batch_images.to(device)

            output = model(
                batch_images
            )


    elapsed = time.time() - start


    print()

    print(
        "Batch size:",
        batch_size
    )

    print(
        "Time:",
        round(elapsed, 4),
        "seconds"
    )
