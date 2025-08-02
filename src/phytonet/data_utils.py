"""Data preparation utilities."""

import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    stratified: bool = True,
):
    """Split dataset into train and validation sets with optional stratification."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output dirs
    for split in ["train", "val"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    if stratified:
        # Collect all images with their class labels
        all_images = []
        class_to_idx = {}

        for idx, class_dir in enumerate(sorted(input_dir.iterdir())):
            if class_dir.is_dir():
                class_to_idx[class_dir.name] = idx
                images = list(class_dir.glob("*"))
                for img_path in images:
                    all_images.append((img_path, idx, class_dir.name))

        # Stratified split
        from sklearn.model_selection import train_test_split

        X = [item[0] for item in all_images]  # image paths
        y = [item[1] for item in all_images]  # class indices

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=train_ratio, random_state=random_seed, stratify=y
        )

        # Create class directories
        for class_name in class_to_idx.keys():
            (output_dir / "train" / class_name).mkdir(parents=True, exist_ok=True)
            (output_dir / "val" / class_name).mkdir(parents=True, exist_ok=True)

        # Copy train images
        train_class_counts = Counter()
        for img_path, class_idx in zip(X_train, y_train):
            class_name = [k for k, v in class_to_idx.items() if v == class_idx][0]
            shutil.copy(img_path, output_dir / "train" / class_name / img_path.name)
            train_class_counts[class_name] += 1

        # Copy val images
        val_class_counts = Counter()
        for img_path, class_idx in zip(X_val, y_val):
            class_name = [k for k, v in class_to_idx.items() if v == class_idx][0]
            shutil.copy(img_path, output_dir / "val" / class_name / img_path.name)
            val_class_counts[class_name] += 1

        # Print statistics
        for class_name in sorted(class_to_idx.keys()):
            train_count = train_class_counts[class_name]
            val_count = val_class_counts[class_name]
            print(f"Class '{class_name}': {train_count} train, {val_count} val")

    else:
        # Original non-stratified split
        for class_dir in input_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*"))
                random.shuffle(images)

                train_count = int(len(images) * train_ratio)
                train_images = images[:train_count]
                val_images = images[train_count:]

                # Create class subdirs
                (output_dir / "train" / class_dir.name).mkdir(
                    parents=True, exist_ok=True
                )
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
