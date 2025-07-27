"""Inference utilities for trained model."""

from pathlib import Path
from typing import List, cast

import torch
from PIL import Image
from tqdm import tqdm

from .model import PhytoplanktonClassifier
from .transforms import get_val_transform


def predict_image(
    image_path: str,
    model_path: str,
    class_names: List[str] | None = None,
    num_classes: int | None = None,
    image_size: int = 224,
) -> tuple:
    """Predict class for a single image."""

    # Load model and get class names from it if not provided
    if class_names is None or num_classes is None:
        # First load to get metadata
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
            num_classes = checkpoint["num_classes"]
        else:
            raise ValueError(
                "Class names not found in model file and not provided as parameter"
            )

    # Type assertion since we know they're not None after the check above
    assert class_names is not None and num_classes is not None
    classifier = PhytoplanktonClassifier(num_classes=num_classes)
    classifier.load_model(model_path)
    classifier.model.eval()

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
    class_names: List[str] | None = None,
    num_classes: int | None = None,
    image_size: int = 224,
    batch_size: int = 32,
) -> List[tuple]:
    """Predict classes for all images in a directory."""

    # Load model and get class names from it if not provided
    if class_names is None or num_classes is None:
        # First load to get metadata
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
            num_classes = checkpoint["num_classes"]
        else:
            raise ValueError(
                "Class names not found in model file and not provided as parameter"
            )

    # Type assertion since we know they're not None after the check above
    assert class_names is not None and num_classes is not None
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

    # Set model to eval mode once
    classifier.model.eval()

    # Process images in batches
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_files = image_files[i : i + batch_size]
        batch_tensors = []

        # Load and preprocess batch of images
        for image_path in batch_files:
            image = Image.open(image_path).convert("RGB")
            tensor = cast(torch.Tensor, transform(image))
            batch_tensors.append(tensor)

        # Stack tensors into batch
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to(classifier.device)

            # Predict on batch
            with torch.no_grad():
                outputs = classifier.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pred_indices = outputs.argmax(dim=1)
                pred_probs = probabilities.gather(1, pred_indices.unsqueeze(1)).squeeze(
                    1
                )

            # Store results
            for j, image_path in enumerate(batch_files):
                pred_idx = pred_indices[j].item()
                pred_prob = pred_probs[j].item()
                predicted_class = class_names[pred_idx]
                results.append((str(image_path), predicted_class, pred_prob))

    return results
