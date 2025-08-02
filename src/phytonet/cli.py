"""Command line interface for PhytoNet."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

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
    parser.add_argument(
        "--loss-type", 
        choices=["cross_entropy", "focal", "weighted_focal"], 
        default="weighted_focal", 
        help="Loss function type (weighted_focal recommended for imbalanced data)"
    )
    parser.add_argument(
        "--use-mixup", action="store_true", default=True,
        help="Enable MixUp/CutMix data augmentation (recommended)"
    )
    parser.add_argument(
        "--no-mixup", dest="use_mixup", action="store_false",
        help="Disable MixUp/CutMix data augmentation"
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.3,
        help="Dropout rate for regularization (default: 0.3)"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=5,
        help="Early stopping patience in epochs (default: 5)"
    )

    args = parser.parse_args()

    # Train model
    _, class_names, _ = train_model(
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        loss_type=args.loss_type,
        use_mixup=args.use_mixup,
        dropout_rate=args.dropout_rate,
        early_stopping_patience=args.early_stopping_patience,
    )

    print(f"Model saved to {args.model_path} with {len(class_names)} classes")


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
        "--output", help="Output file for batch predictions (CSV format)"
    )

    args = parser.parse_args()

    # Load class names if provided, otherwise they'll be loaded from model
    class_names = None
    if args.classes_path:
        try:
            with open(args.classes_path, "r") as f:
                class_names = json.load(f)
        except FileNotFoundError:
            print(f"Class names file not found: {args.classes_path}")
            return

    input_path = Path(args.input)

    if input_path.is_file():
        # Single image prediction
        predicted_class, confidence = predict_image(
            str(input_path),
            args.model_path,
            class_names,
            args.num_classes,
            args.image_size,
        )
        print(f"Predicted class: {predicted_class} (confidence: {confidence:.3f})")

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
            # Validate output file extension
            if not args.output.endswith(".csv"):
                print("Error: Output file must have .csv extension")
                return

            # Ensure the output directory exists
            output_dir = Path(args.model_path).parent

            # Should never happen, but just in case
            if not output_dir.exists():
                print(f"Creating output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(args.output).name
            outfile = output_dir / filename

            # Convert results to DataFrame for saving
            df = pd.DataFrame(results)
            df.columns = ["image_path", "predicted_class", "probability"]

            print("Batch predictions completed. Here are the first few results:\n")
            print(df.head(n=5).to_string(index=False))

            if os.path.exists(outfile):
                print(f"\nOutput file already exists: {outfile}. Overwriting...")
            else:
                print(f"\nSaving results to {outfile}...")

            df.to_csv(outfile, index=False)
        else:
            # Print to console
            print("Batch predictions completed:\n")
            for image_path, predicted_class, confidence in results:
                print(f"{image_path}: {predicted_class} (confidence: {confidence:.3f})")

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
