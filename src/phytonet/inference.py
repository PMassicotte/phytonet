"""Inference utilities for trained model."""

from pathlib import Path
from typing import List, cast

import pandas as pd
import torch
from PIL import Image

from .model import PhytoplanktonClassifier
from .transforms import get_val_transform


def predict_image(
    image_path: str,
    model_path: str,
    class_names: List[str],
    num_classes: int = 40,
    image_size: int = 224,
) -> tuple:
    """Predict class for a single image."""

    # Load model
    classifier = PhytoplanktonClassifier(num_classes=num_classes)
    classifier.load_model(model_path)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transform(image_size)
    tensor = cast(torch.Tensor, transform(image))
    input_tensor = tensor.unsqueeze(0)

    # Predict
    pred_idx, pred_prob = classifier.predict(input_tensor)
    predicted_class = class_names[pred_idx]

    return predicted_class, pred_prob


def batch_predict(
    image_dir: str,
    model_path: str,
    class_names: List[str],
    output_file: str = "predictions.parquet",
    num_classes: int = 40,
    image_size: int = 224,
) -> List[tuple]:
    """Predict classes for all images in a directory and save to Parquet."""

    # Load model
    classifier = PhytoplanktonClassifier(num_classes=num_classes)
    classifier.load_model(model_path)

    # Get all image files
    image_dir_path = Path(image_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        f for f in image_dir_path.iterdir() if f.suffix.lower() in image_extensions
    ]

    results = []
    transform = get_val_transform(image_size)

    for image_path in image_files:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        tensor = cast(torch.Tensor, transform(image))
        input_tensor = tensor.unsqueeze(0)

        # Predict
        pred_idx, pred_prob = classifier.predict(input_tensor)
        predicted_class = class_names[pred_idx]

        results.append((str(image_path), predicted_class, pred_prob))

    # Save results to Parquet and CSV
    df = pd.DataFrame(results, columns=["image_path", "predicted_class", "probability"])  # type: ignore[arg-type]
    df.to_parquet(output_file, index=False)
    df.to_csv(output_file.replace(".parquet", ".csv"), index=False)

    return results
