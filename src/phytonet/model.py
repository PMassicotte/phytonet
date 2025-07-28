"""Model definition and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score
from timm import create_model


class PhytoplanktonClassifier:
    """Phytoplankton classifier using EfficientNetV2."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "tf_efficientnetv2_s",
        class_weights: torch.Tensor | None = None,
    ):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Loss and optimizer with class weights for imbalance
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        train_loss, train_correct = 0, 0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

            # Collect predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate additional metrics
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        return train_loss, train_correct, macro_f1, balanced_acc

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

                # Collect predictions and labels for metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate additional metrics
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)

        return val_loss, val_correct, macro_f1, balanced_acc

    def save_model(self, path: str, class_names: list[str] | None = None):
        """Save model state dict and metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
        }
        if class_names is not None:
            checkpoint['class_names'] = class_names
        torch.save(checkpoint, path)

    def load_model(self, path: str):
        """Load model state dict and metadata."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return checkpoint.get('class_names')
        else:
            # Legacy format - just state dict
            self.model.load_state_dict(checkpoint)
            return None

    def predict(self, input_tensor):
        """Make prediction on input tensor."""
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            probabilities = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            pred_prob = probabilities[0][pred_idx].item()
        return pred_idx, pred_prob

    def step_scheduler(self, val_loss):
        """Step the learning rate scheduler."""
        self.scheduler.step(val_loss)
