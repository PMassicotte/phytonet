"""Compute dataset-specific normalization values for grayscale phytoplankton images."""

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .transforms import ResizeKeepAspectPad


class NormalizationDataset(Dataset):
    """Dataset for computing normalization statistics."""

    def __init__(self, data_dir, image_size=224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.image_paths = []

        # Collect all image paths
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.png"):
                    self.image_paths.append(img_path)

        self.transform = transforms.Compose(
            [
                ResizeKeepAspectPad(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert grayscale to RGB
        return self.transform(image)


def compute_normalization_stats(data_dir, image_size=224, batch_size=64):
    """Compute mean and std for dataset normalization."""

    dataset = NormalizationDataset(data_dir, image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Computing normalization stats for {len(dataset)} images...")

    # Compute mean
    mean = torch.zeros(3)
    total_samples = 0

    for batch in tqdm(dataloader, desc="Computing mean"):
        batch_samples = batch.size(0)
        batch = batch.view(batch_samples, batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples

    # Compute std
    var = torch.zeros(3)
    total_samples = 0

    for batch in tqdm(dataloader, desc="Computing std"):
        batch_samples = batch.size(0)
        batch = batch.view(batch_samples, batch.size(1), -1)
        var += ((batch - mean.unsqueeze(1)) ** 2).sum([0, 2])
        total_samples += batch_samples * batch.size(2)

    std = torch.sqrt(var / total_samples)

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute dataset normalization statistics"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Path to training data directory"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Image size")

    args = parser.parse_args()

    mean, std = compute_normalization_stats(args.data_dir, args.image_size)

    print(f"\nDataset normalization statistics:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"\nTo use in transforms:")
    print(f"transforms.Normalize({mean}, {std})")
