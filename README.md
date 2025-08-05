# PhytoNet <img src="logo.png" align="right" height="138" alt="PhytoNet logo" /></a>

A Python package for phytoplankton classification using deep learning.

## Installation

Install using uv:

```bash
uv sync
```

Or install in development mode:

```bash
uv sync --group dev
```

## Usage

### Training

Train a model on your phytoplankton dataset:

```bash
uv run phytonet-train --data-dir ./data/ --input-dir ./data/original/ --epochs 20
```

The training process will:

- Automatically split your dataset into train/validation sets
- Calculate the number of classes from your input directory structure
- Save the trained model with embedded class names and metadata

### New cli

This now automatically uses:

- ✅ Weighted Focal Loss (best for 180:1 class imbalance)
- ✅ MixUp/CutMix (30% each, prevents overfitting)
- ✅ Dropout 0.3 (regularization)
- ✅ Early stopping (patience=5)
- ✅ Smoothed class weights (sqrt scaling)
- ✅ Stratified splitting (preserves class distribution)

Optional Overrides:

# Disable MixUp if you want faster training

```bash
uv run phytonet-train --epochs 20 --no-mixup
```

# Use different loss function

```bash
uv run phytonet-train --epochs 20 --loss-type focal
```

# Adjust dropout

```bash
uv run phytonet-train --epochs 20 --dropout-rate 0.5
```

# More patient early stopping

```bash
uv run phytonet-train --epochs 20 --early-stopping-patience 8
```

### Prediction

Make predictions on new images:

```bash
# Single image prediction
uv run phytonet-predict image.png --model-path ifcb_model.pth

# Batch prediction on directory
uv run phytonet-predict ./test_images/ --model-path ifcb_model.pth --output predictions.csv

# Batch prediction on a directory with a specific model that now includes class names
uv run phytonet-predict ~/Downloads --model-path data/best_model_20250725_173054_epoch12_acc0.95.pth --output predictions.csv
```

## Development

Install development dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Format code:

```bash
uv run black src/
uv run isort src/
```

## Features

- **Custom transforms**: Maintains aspect ratio while resizing images
- **Flexible training**: Configurable hyperparameters and model architecture
- **Easy inference**: Single image and batch prediction support
- **CLI tools**: Simple command-line interface for training and prediction
- **Proper packaging**: Uses modern Python packaging with pyproject.toml
