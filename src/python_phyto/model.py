"""Model definition and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from timm import create_model


class PhytoplanktonClassifier:
    """Phytoplankton classifier using EfficientNetV2."""

    def __init__(self, num_classes: int = 40, model_name: str = "tf_efficientnetv2_s"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss, train_correct = 0, 0

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

        return train_loss, train_correct

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss, val_correct = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        return val_loss, val_correct

    def save_model(self, path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, input_tensor):
        """Make prediction on input tensor."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))
            pred = output.argmax(dim=1).item()
        return pred

