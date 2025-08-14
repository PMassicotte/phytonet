# PhytoNet <img src="logo.png" align="right" height="138" alt="PhytoNet logo" /></a>

`PhytoNet` is a tool that uses deep learning to identify different types of phytoplankton (microscopic marine plants) from images.

## What does `PhytoNet` do?

- **Identifies phytoplankton species** from microscopic images automatically
- **Learns from your data** - train custom models on your specific organism types
- **Handles large datasets** - process thousands of images in minutes
- **Works with imbalanced data** - handles datasets where some species are much rarer than others

## Quick start

### 1. Installation

```bash
uv sync
```

### Getting help

There are two main commands you will use to train and predict with `PhytoNet`. To see the available options for each command, run:

```bash
uv run phytonet-train --help
uv run phytonet-predict --help
```

### 2. Train a Model

Train `PhytoNet` to recognize your phytoplankton species:

```bash
uv run phytonet-train --data-dir ./data/ --input-dir ./data/original/ --epochs 20
```

**What this does:**

- Automatically organizes your images into training and validation sets (80% train, 20% val)
- Learns to distinguish between different species in your dataset
- Saves a trained model that remembers what it learned

After training, you will find your model saved in the `data/models/` directory with a timestamped folder name like `YYYYMMDD-HHMMSS/`. Inside, you'll find:

- `best_model_*.pth`: The best model weights
- `training_history_*.csv`: Training metrics like accuracy and loss
- `learning_rate_*.png`: Learning rate schedule plot

### 3. Make predictions

Use your trained model to identify new organisms:

```bash
# Classify a single image
uv run phytonet-predict image.png --model-path your_model.pth

# Classify many images at once
uv run phytonet-predict ./new_images/ --model-path your_model.pth --output results.csv
```

The `--output` option saves the predictions to a CSV file with species names and confidence scores in the same directory as the model.

After running the prediction command, you will see output like this in the model directory (depending on the --output option). For example, if you transforms

```bash
uv run phytonet-predict ./new_images/ --model-path 20250729-120937/best_model_20250729_141727_epoch14_acc0.96.pth --output results.csv
```

This will create a `results.csv` file inside the `20250729-120937/` directory with content like:

```
image_path,predicted_class,probability
new_images/chaet_001.jpg,Chaetoceros,0.98
new_images/diat_002.png,Diatoma,0.95
new_images/skel_003.jpg,Skeletonema,0.99
```

**CSV format details:**

- `image_path`: Relative path to the input image
- `predicted_class`: Phytoplankton species classification (one of the trained species found in `best_model_20250729_141727_epoch14_acc0.96.pth`)
- `probability`: Confidence score (0.0 to 1.0)

## Advanced training options

`PhytoNet` automatically uses state-of-the-art techniques, but you can customize the training:

```bash
# Faster training (disable data mixing)
uv run phytonet-train --epochs 20 --no-mixup

# More aggressive regularization
uv run phytonet-train --epochs 20 --dropout-rate 0.5

# More patient training (wait longer before stopping)
uv run phytonet-train --epochs 20 --early-stopping-patience 8
```

## Features

- **Custom transforms**: Maintains aspect ratio while resizing images
- **Flexible training**: Configurable hyperparameters and model architecture
- **Easy inference**: Single image and batch prediction support
- **CLI tools**: Simple command-line interface for training and prediction
- **Proper packaging**: Uses modern Python packaging with `pyproject.toml`

## Data organization

### Training data structure

Organize your training images in folders by species name. Each folder should contain all images of that species:

```
data/original/
├── Chaetoceros/
│   ├── chaet_001.jpg
│   ├── chaet_002.png
│   └── chaet_003.jpg
├── Diatoma/
│   ├── diat_001.jpg
│   └── diat_002.png
└── Skeletonema/
    ├── skel_001.jpg
    ├── skel_002.jpg
    └── skel_003.png
```

### What `PhytoNet` creates

After training, `PhytoNet` automatically organizes your data:

```
data/
├── original/       # Your input images (organized by species)
│   ├── alexandrium_catenella/
│   ├── chaetoceros/
│   └── other_species/
├── train/          # Training split (created automatically)
│   ├── alexandrium_catenella/
│   ├── chaetoceros/
│   └── other_species/
├── val/            # Validation split (created automatically)
│   ├── alexandrium_catenella/
│   ├── chaetoceros/
│   └── other_species/
└── models/         # Saved trained models
    └── YYYYMMDD-HHMMSS/
        ├── best_model_*.pth
        ├── training_history_*.csv
        └── learning_rate_*.png
```

**Key features:**

- **Any folder structure**: `PhytoNet` recursively scans all subdirectories
- **Mixed formats**: Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` in same directory
- **Flexible naming**: No specific file naming requirements
- **Batch processing**: Processes thousands of images efficiently

### Important guidelines

- **Folder names = Species labels**: Use clear, consistent species names as `PhytoNet` uses folder names as class labels
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`
- **Minimum images**: At least 10-20 images per species for effective training
- **Image quality**: Use high-resolution images (>224x224 pixels recommended)
- **Consistent naming**: Use systematic file naming for easier organization
