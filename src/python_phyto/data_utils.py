"""Data preparation utilities."""

import random
import shutil
from pathlib import Path


def split_dataset(
    input_dir: Path, output_dir: Path, train_ratio: float = 0.8, random_seed: int = 42
):
    """Split dataset into train and validation sets."""
    random.seed(random_seed)

    # Create output dirs
    for split in ["train", "val"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # For each class
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*"))
            random.shuffle(images)

            train_count = int(len(images) * train_ratio)
            train_images = images[:train_count]
            val_images = images[train_count:]

            # Create class subdirs
            (output_dir / "train" / class_dir.name).mkdir(parents=True, exist_ok=True)
            (output_dir / "val" / class_dir.name).mkdir(parents=True, exist_ok=True)

            # Copy images
            for img_path in train_images:
                shutil.copy(
                    img_path, output_dir / "train" / class_dir.name / img_path.name
                )

            for img_path in val_images:
                shutil.copy(
                    img_path, output_dir / "val" / class_dir.name / img_path.name
                )

            print(
                f"Class '{class_dir.name}': {len(train_images)} train, {len(val_images)} val"
            )

    print(
        f"\nDone! Data split into '{output_dir / 'train'}' and '{output_dir / 'val'}'"
    )
