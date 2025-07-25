"""Training script for phytoplankton classification."""

import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .data_utils import split_dataset
from .model import PhytoplanktonClassifier
from .transforms import get_train_transform, get_val_transform


def train_model(
    data_dir: str = "./data/",
    input_dir: str = "./data/original/",
    batch_size: int = 32,
    num_epochs: int = 10,
    num_classes: int = 40,
    image_size: int = 224,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    model_save_path: str = "ifcb_model.pt",
):
    """Train the phytoplankton classifier."""

    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(data_dir)

    # Split dataset if needed
    if not (output_path / "train").exists() or not (output_path / "val").exists():
        print("Splitting dataset...")
        split_dataset(input_path, output_path, train_ratio, random_seed)

    # Setup transforms
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    # Setup datasets and loaders
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    classifier = PhytoplanktonClassifier(num_classes=num_classes)

    # Track best validation accuracy
    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_correct = classifier.train_epoch(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"), epoch
        )

        # Validate
        val_loss, val_correct = classifier.validate(val_loader)

        # Calculate accuracies
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)

        # Print metrics
        print(
            f"Epoch {epoch + 1}: "
            f"Train loss: {train_loss / len(train_ds):.4f}, "
            f"acc: {train_acc:.4f} | "
            f"Val loss: {val_loss / len(val_ds):.4f}, "
            f"acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            classifier.save_model(model_save_path)
            print(f"New best model saved! Val acc: {val_acc:.4f}")

    print(
        f"Training completed. Best model from epoch {best_epoch} with val acc: {best_val_acc:.4f}"
    )

    return classifier, train_ds.classes


if __name__ == "__main__":
    train_model()
