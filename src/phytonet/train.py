"""Training script for phytoplankton classification."""

import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .data_utils import split_dataset
from .model import PhytoplanktonClassifier
from .training_history import TrainingHistory
from .transforms import get_train_transform, get_val_transform


def train_model(
    data_dir: str = "./data/",
    input_dir: str = "./data/original/",
    batch_size: int = 32,
    num_epochs: int = 10,
    image_size: int = 224,
    train_ratio: float = 0.8,
    random_seed: int = 42,
):
    """Train the phytoplankton classifier."""

    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(data_dir)

    # Determine number of classes dynamically
    num_classes = len([d for d in input_path.iterdir() if d.is_dir()])

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

    # Calculate class weights for imbalanced dataset
    class_counts = Counter(train_ds.targets)
    total_samples = len(train_ds)
    class_weights = torch.tensor(
        [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
    )

    print(f"Class distribution: {dict(sorted(class_counts.items()))}")
    print(f"Class weights computed for {num_classes} classes")

    # Initialize model with class weights
    classifier = PhytoplanktonClassifier(
        num_classes=num_classes, class_weights=class_weights
    )

    # Create models directory with date
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    models_save_dir = output_path / f"models/{date_str}"
    models_save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training history tracker
    history = TrainingHistory(save_dir=str(models_save_dir))
    history.update_metadata(
        model_name=classifier.model_name,
        num_classes=num_classes,
        batch_size=batch_size,
        total_epochs=num_epochs,
        image_size=image_size,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    # Track best validation accuracy
    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_correct, train_macro_f1, train_balanced_acc = (
            classifier.train_epoch(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            )
        )

        # Validate
        val_loss, val_correct, val_macro_f1, val_balanced_acc = classifier.validate(
            val_loader
        )

        # Calculate accuracies
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)

        # Get current learning rate
        current_lr = classifier.optimizer.param_groups[0]["lr"]

        # Add epoch metrics to history
        history.add_epoch(
            epoch=epoch + 1,
            train_loss=train_loss / len(train_ds),
            train_accuracy=train_acc,
            train_macro_f1=train_macro_f1,
            train_balanced_accuracy=train_balanced_acc,
            val_loss=val_loss / len(val_ds),
            val_accuracy=val_acc,
            val_macro_f1=val_macro_f1,
            val_balanced_accuracy=val_balanced_acc,
            learning_rate=current_lr,
        )

        # Step scheduler
        classifier.step_scheduler(val_loss / len(val_ds))

        # Print metrics
        print(
            f"Epoch {epoch + 1}: "
            f"Train loss: {train_loss / len(train_ds):.4f}, "
            f"acc: {train_acc:.4f}, macro_f1: {train_macro_f1:.4f}, bal_acc: {train_balanced_acc:.4f} | "
            f"Val loss: {val_loss / len(val_ds):.4f}, "
            f"acc: {val_acc:.4f}, macro_f1: {val_macro_f1:.4f}, bal_acc: {val_balanced_acc:.4f}"
        )

        # Save best model (using balanced accuracy as primary metric for imbalanced data)
        if val_balanced_acc > best_val_acc:
            date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = (
                models_save_dir
                / f"best_model_{date_time_str}_epoch{epoch}_acc{val_balanced_acc:.2f}.pth"
            )

            best_val_acc = val_balanced_acc
            best_epoch = epoch + 1
            classifier.save_model(str(model_save_path), train_ds.classes)
            history.set_best_model(epoch + 1, "val_balanced_accuracy", val_balanced_acc)

            print(f"New best model saved! Val balanced acc: {val_balanced_acc:.4f}")

    print(
        f"Training completed. Best model from epoch {best_epoch} with val balanced acc: {best_val_acc:.4f}"
    )

    # Save training history
    history.save_csv()

    # Generate and save plots
    history.plot_metrics(show=False)
    history.plot_learning_rate(show=False)

    # Print training summary
    history.print_summary()

    return classifier, train_ds.classes, history


if __name__ == "__main__":
    train_model()
