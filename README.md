# PhytoNet

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

### Prediction

Make predictions on new images:

```bash
# Single image
uv run phytonet-predict image.png --model-path ifcb_model.pt

# Batch prediction on directory
uv run phytonet-predict ./test_images/ --model-path ifcb_model.pt --output results.json

# Override with custom class names file if needed
uv run phytonet-predict image.png --model-path ifcb_model.pt --classes-path custom_classes.json

# Example:
uv run phytonet-predict ./data/original/alexandrium_catenella/D20220813T053409_IFCB145_00023.png --model-path ifcb_model.pt --classes-path classes.json

uv run phytonet-predict /media/work/others/mathieu_ardyna/ifcb/ifcb_classifier/run-data/02_Greenedge_Cruise_2016/ --model-path ifcb_model.pt --classes-path classes.json

# Batch prediction on a directory with a specific model that now includes class names
uv run phytonet-predict ~/Downloads --model-path data/best_model_20250725_173054_epoch12_acc0.95.pth
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

## Project Structure

<!-- TODO: redo it -->

```
src/phytonet/
├── __init__.py          # Package initialization
├── model.py             # Model definition and training utilities
├── transforms.py        # Image preprocessing transforms
├── data_utils.py        # Dataset splitting utilities
├── train.py             # Training script
├── inference.py         # Inference utilities
└── cli.py              # Command line interface
```

## Features

- **Custom transforms**: Maintains aspect ratio while resizing images
- **Flexible training**: Configurable hyperparameters and model architecture
- **Easy inference**: Single image and batch prediction support
- **CLI tools**: Simple command-line interface for training and prediction
- **Proper packaging**: Uses modern Python packaging with pyproject.toml

