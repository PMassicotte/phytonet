"""Command line interface for PhytoNet."""

import argparse
import json
from pathlib import Path

from .inference import batch_predict, predict_image
from .train import train_model


def train_cli():
    """CLI for training the model."""
    parser = argparse.ArgumentParser(description="Train phytoplankton classifier")
    parser.add_argument(
        "--data-dir", default="./data/", help="Directory containing train/val splits"
    )
    parser.add_argument(
        "--input-dir",
        default="./data/original/",
        help="Directory containing original data to split",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--num-classes", type=int, default=40, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--model-path", default="ifcb_model.pt", help="Path to save trained model"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Ratio of data for training"
    )

    args = parser.parse_args()

    # Train model
    classifier, class_names = train_model(
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_classes=args.num_classes,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        model_save_path=args.model_path,
    )

    # Save class names
    class_names_path = Path(args.model_path).with_suffix(".json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")


def predict_cli():
    """CLI for making predictions."""
    parser = argparse.ArgumentParser(description="Predict phytoplankton classes")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument(
        "--classes-path",
        help="Path to class names JSON file (auto-detected if not provided)",
    )
    parser.add_argument("--num-classes", type=int, default=40, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--output", help="Output file for batch predictions (JSON format)"
    )

    args = parser.parse_args()

    # Load class names
    if args.classes_path:
        classes_path = args.classes_path
    else:
        classes_path = Path(args.model_path).with_suffix(".json")

    try:
        with open(classes_path, "r") as f:
            class_names = json.load(f)
    except FileNotFoundError:
        print(f"Class names file not found: {classes_path}")
        print(
            "Please provide --classes-path or ensure the .json file exists alongside the model"
        )
        return

    input_path = Path(args.input)

    if input_path.is_file():
        # Single image prediction
        predicted_class = predict_image(
            str(input_path),
            args.model_path,
            class_names,
            args.num_classes,
            args.image_size,
        )
        print(f"Predicted class: {predicted_class}")

    elif input_path.is_dir():
        # Batch prediction
        results = batch_predict(
            str(input_path),
            args.model_path,
            class_names,
            args.num_classes,
            args.image_size,
        )

        if args.output:
            # Save to JSON file
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            # Print to console
            for image_path, predicted_class in results:
                print(f"{image_path}: {predicted_class}")

    else:
        print(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_cli()
    else:
        print("Usage: python -m phytonet.cli [train|predict] [args...]")

