"""Training history tracker for monitoring and visualizing training metrics."""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


class TrainingHistory:
    """Tracks and manages training history with comprehensive metrics."""

    def __init__(self, save_dir: Optional[str] = None):
        """Initialize training history tracker.

        Args:
            save_dir: Directory to save history files. If None, uses current directory.
        """
        self.save_dir = Path(save_dir) if save_dir else Path(".")
        self.save_dir.mkdir(exist_ok=True)

        # Initialize history storage
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_f1": [],
            "train_balanced_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_balanced_accuracy": [],
            "learning_rate": [],
            "timestamp": [],
        }

        # Metadata
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "model_name": None,
            "num_classes": None,
            "batch_size": None,
            "total_epochs": None,
            "best_epoch": None,
            "best_val_metric": None,
            "best_val_value": None,
        }

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        train_macro_f1: float,
        train_balanced_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        val_macro_f1: float,
        val_balanced_accuracy: float,
        learning_rate: float,
    ):
        """Add metrics for one epoch to history.

        Args:
            epoch: Epoch number (1-indexed)
            train_loss: Training loss
            train_accuracy: Training accuracy
            train_macro_f1: Training macro F1 score
            train_balanced_accuracy: Training balanced accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            val_macro_f1: Validation macro F1 score
            val_balanced_accuracy: Validation balanced accuracy
            learning_rate: Current learning rate
        """
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["train_macro_f1"].append(train_macro_f1)
        self.history["train_balanced_accuracy"].append(train_balanced_accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["val_macro_f1"].append(val_macro_f1)
        self.history["val_balanced_accuracy"].append(val_balanced_accuracy)
        self.history["learning_rate"].append(learning_rate)
        self.history["timestamp"].append(datetime.now().isoformat())

    def update_metadata(self, **kwargs):
        """Update metadata with additional information."""
        self.metadata.update(kwargs)

    def set_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Record information about the best model."""
        self.metadata["best_epoch"] = epoch
        self.metadata["best_val_metric"] = metric_name
        self.metadata["best_val_value"] = metric_value

    def get_dataframe(self) -> pd.DataFrame:
        """Get history as pandas DataFrame."""
        return pd.DataFrame(self.history)

    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics across all epochs."""
        if not self.history["epoch"]:
            return {}

        df = self.get_dataframe()
        return {
            "best_train_accuracy": {
                "value": df["train_accuracy"].max(),
                "epoch": df.loc[df["train_accuracy"].idxmax(), "epoch"],
            },
            "best_val_accuracy": {
                "value": df["val_accuracy"].max(),
                "epoch": df.loc[df["val_accuracy"].idxmax(), "epoch"],
            },
            "best_train_balanced_accuracy": {
                "value": df["train_balanced_accuracy"].max(),
                "epoch": df.loc[df["train_balanced_accuracy"].idxmax(), "epoch"],
            },
            "best_val_balanced_accuracy": {
                "value": df["val_balanced_accuracy"].max(),
                "epoch": df.loc[df["val_balanced_accuracy"].idxmax(), "epoch"],
            },
            "best_train_macro_f1": {
                "value": df["train_macro_f1"].max(),
                "epoch": df.loc[df["train_macro_f1"].idxmax(), "epoch"],
            },
            "best_val_macro_f1": {
                "value": df["val_macro_f1"].max(),
                "epoch": df.loc[df["val_macro_f1"].idxmax(), "epoch"],
            },
            "lowest_train_loss": {
                "value": df["train_loss"].min(),
                "epoch": df.loc[df["train_loss"].idxmin(), "epoch"],
            },
            "lowest_val_loss": {
                "value": df["val_loss"].min(),
                "epoch": df.loc[df["val_loss"].idxmin(), "epoch"],
            },
        }

    def plot_metrics(self, save_path: Optional[str] = None, show: bool = True):
        """Plot training metrics.

        Args:
            save_path: Path to save the plot. If None, auto-generates filename.
            show: Whether to display the plot.
        """
        if not self.history["epoch"]:
            print("No training history to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Training History", fontsize=16)

        epochs = self.history["epoch"]

        # Loss plot
        axes[0, 0].plot(
            epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2
        )
        axes[0, 0].plot(
            epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=2
        )
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(
            epochs, self.history["train_accuracy"], "b-", label="Train Acc", linewidth=2
        )
        axes[0, 1].plot(
            epochs, self.history["val_accuracy"], "r-", label="Val Acc", linewidth=2
        )
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Balanced Accuracy plot
        axes[1, 0].plot(
            epochs,
            self.history["train_balanced_accuracy"],
            "b-",
            label="Train Bal Acc",
            linewidth=2,
        )
        axes[1, 0].plot(
            epochs,
            self.history["val_balanced_accuracy"],
            "r-",
            label="Val Bal Acc",
            linewidth=2,
        )
        axes[1, 0].set_title("Balanced Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Balanced Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Macro F1 plot
        axes[1, 1].plot(
            epochs,
            self.history["train_macro_f1"],
            "b-",
            label="Train Macro F1",
            linewidth=2,
        )
        axes[1, 1].plot(
            epochs,
            self.history["val_macro_f1"],
            "r-",
            label="Val Macro F1",
            linewidth=2,
        )
        axes[1, 1].set_title("Macro F1 Score")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Macro F1")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.save_dir / f"training_history_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_rate(self, save_path: Optional[str] = None, show: bool = True):
        """Plot learning rate schedule.

        Args:
            save_path: Path to save the plot. If None, auto-generates filename.
            show: Whether to display the plot.
        """
        if not self.history["epoch"]:
            print("No training history to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.history["epoch"], self.history["learning_rate"], "g-", linewidth=2
        )
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.save_dir / f"learning_rate_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning rate plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_csv(self, filename: Optional[str] = None):
        """Save history to CSV file.

        Args:
            filename: CSV filename. If None, auto-generates filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}.csv"

        csv_path = self.save_dir / filename
        df = self.get_dataframe()
        df.to_csv(csv_path, index=False)
        print(f"History saved to CSV: {csv_path}")

    @classmethod
    def load_from_pickle(cls, filepath: str) -> "TrainingHistory":
        """Load TrainingHistory object from pickle file.

        Args:
            filepath: Path to pickle file.

        Returns:
            TrainingHistory object.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def print_summary(self):
        """Print a summary of the training history."""
        if not self.history["epoch"]:
            print("No training history available.")
            return

        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        # Metadata
        print(f"Model: {self.metadata.get('model_name', 'Unknown')}")
        print(f"Classes: {self.metadata.get('num_classes', 'Unknown')}")
        print(f"Batch Size: {self.metadata.get('batch_size', 'Unknown')}")
        print(f"Total Epochs: {len(self.history['epoch'])}")
        print(f"Start Time: {self.metadata.get('start_time', 'Unknown')}")

        if self.metadata.get("best_epoch"):
            print(
                f"Best Model: Epoch {self.metadata['best_epoch']} "
                f"({self.metadata['best_val_metric']}: {self.metadata['best_val_value']:.4f})"
            )

        print("\n" + "-" * 60)
        print("BEST METRICS")
        print("-" * 60)

        best_metrics = self.get_best_metrics()
        for metric_name, metric_info in best_metrics.items():
            print(
                f"{metric_name:25}: {metric_info['value']:.4f} (Epoch {metric_info['epoch']})"
            )

        print("\n" + "-" * 60)
        print("FINAL EPOCH METRICS")
        print("-" * 60)

        # Final epoch metrics
        final_idx = -1
        print(f"Train Loss:           {self.history['train_loss'][final_idx]:.4f}")
        print(f"Train Accuracy:       {self.history['train_accuracy'][final_idx]:.4f}")
        print(
            f"Train Balanced Acc:   {self.history['train_balanced_accuracy'][final_idx]:.4f}"
        )
        print(f"Train Macro F1:       {self.history['train_macro_f1'][final_idx]:.4f}")
        print(f"Val Loss:             {self.history['val_loss'][final_idx]:.4f}")
        print(f"Val Accuracy:         {self.history['val_accuracy'][final_idx]:.4f}")
        print(
            f"Val Balanced Acc:     {self.history['val_balanced_accuracy'][final_idx]:.4f}"
        )
        print(f"Val Macro F1:         {self.history['val_macro_f1'][final_idx]:.4f}")
        print(f"Learning Rate:        {self.history['learning_rate'][final_idx]:.2e}")

        print("=" * 60)
