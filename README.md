# PartFlow3D

Experimental playground for learning 3D data processing and neural network-based segmentation, with a focus on understanding challenges relevant to dental design workflows.

## Project Structure

```
PartFlow3D/
├── configs/          # Configuration files for models and training
├── data/
│   ├── raw/         # Raw downloaded data
│   └── processed/   # Preprocessed data ready for training
├── models/          # Saved model checkpoints
├── notebooks/       # Jupyter notebooks for exploration
├── scripts/         # Utility scripts (download, preprocessing, etc.)
└── src/            # Source code for models and training
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env with your preferences
```

### 4. Download Dataset

Download the PartObjaverse-Tiny dataset (3D part segmentation data):

```bash
# Download entire dataset
python scripts/download_data.py --output-dir data/raw

# Or download specific subset
python scripts/download_data.py --output-dir data/raw --subset train
```

## Dataset

**PartObjaverse-Tiny**: A 3D part segmentation dataset containing meshes with semantic part labels. This dataset is ideal for learning 3D segmentation tasks similar to those used in dental design workflows.

- Dataset: [yhyang-myron/PartObjaverse-Tiny](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny)
- Paper: [SAMPart3D](https://yhyang-myron.github.io/SAMPart3D-website/?utm_source=chatgpt.com)

## Usage

### 1. Preprocess Data

Convert raw meshes to point clouds:

```bash
python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed
```

### 2. Train Model

Train PointNet for semantic segmentation:

```bash
# Train with default settings
python scripts/train.py

# Custom training
python scripts/train.py --batch-size 16 --num-epochs 200 --lr 0.0005

# Monitor training
tensorboard --logdir runs/
```

### 3. Evaluate Model

Evaluate trained model on test set:

```bash
python scripts/evaluate.py

# Metrics computed: accuracy, per-class IoU, mean IoU, confusion matrix
```

### 4. Visualize Predictions

Generate interactive 3D visualizations:

```bash
# Visualize random samples
python scripts/visualize_predictions.py --num-samples 5

# Show prediction errors
python scripts/visualize_predictions.py --show-errors
```

## Current Features

- ✓ **Data Pipeline**: Download, preprocess, and augment 3D point cloud data
- ✓ **PointNet Model**: Semantic segmentation with feature transforms
- ✓ **Training**: TensorBoard logging, checkpointing, learning rate scheduling
- ✓ **Evaluation**: Accuracy, IoU metrics, confusion matrix
- ✓ **Visualization**: Interactive 3D plots of predictions vs ground truth

## Goals

- Understand 3D data formats (meshes, point clouds, voxels)
- Learn preprocessing pipelines for 3D data
- Experiment with neural networks for 3D segmentation
- Explore trade-offs in model architecture, data representation, and performance
- Gain practical experience relevant to dental design workflows at companies like cadflow.ai