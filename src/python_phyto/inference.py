"""Inference utilities for trained model."""

from pathlib import Path
from typing import List

from PIL import Image

from .model import PhytoplanktonClassifier
from .transforms import get_val_transform


def predict_image(
    image_path: str,
    model_path: str,
    class_names: List[str],
    num_classes: int = 40,
    image_size: int = 224,
) -> str:
    """Predict class for a single image."""

    # Load model
    classifier = PhytoplanktonClassifier(num_classes=num_classes)
    classifier.load_model(model_path)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transform(image_size)
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    pred_idx = classifier.predict(input_tensor)
    predicted_class = class_names[pred_idx]

    return predicted_class


def batch_predict(
    image_dir: str,
    model_path: str,
    class_names: List[str],
    num_classes: int = 40,
    image_size: int = 224,
) -> List[tuple]:
    """Predict classes for all images in a directory."""

    # Load model
    classifier = PhytoplanktonClassifier(num_classes=num_classes)
    classifier.load_model(model_path)

    # Get all image files
    image_dir = Path(image_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    results = []
    transform = get_val_transform(image_size)

    for image_path in image_files:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        pred_idx = classifier.predict(input_tensor)
        predicted_class = class_names[pred_idx]

        results.append((str(image_path), predicted_class))

    return results

