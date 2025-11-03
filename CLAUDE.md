# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PartFlow3D is an experimental playground for learning 3D data processing and neural network-based segmentation. The primary goal is to gain practical experience with 3D segmentation workflows relevant to dental design applications (e.g., cadflow.ai).

## Setup Commands

```bash
# Initial environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Configure environment variables
cp .env.example .env
# Edit .env with custom paths/hyperparameters if needed
```

## Data Pipeline

### 1. Download Dataset

The project uses PartObjaverse-Tiny, a 3D part segmentation dataset from HuggingFace containing meshes with semantic and instance part labels.

```bash
# Download and extract all data (recommended for first-time setup)
python scripts/download_data.py --output-dir data/raw

# Download without extracting (if you need zip files only)
python scripts/download_data.py --output-dir data/raw --no-extract

# Keep zip files after extraction (default: removes them)
python scripts/download_data.py --output-dir data/raw --keep-zip
```

The download script fetches three zip files:
- `PartObjaverse-Tiny_mesh.zip` - 3D mesh data (.glb files)
- `PartObjaverse-Tiny_semantic_gt.zip` - Semantic segmentation labels (per-face)
- `PartObjaverse-Tiny_instance_gt.zip` - Instance segmentation labels (per-face)

### 2. Preprocess Dataset

Convert meshes to point clouds for neural network training:

```bash
# Preprocess with default settings (2048 points, normalized)
python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed

# Use more points for higher resolution
python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed --num-points 4096

# Disable normalization
python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed --no-normalize
```

**What preprocessing does:**
- Loads GLB meshes (handles Scene objects with multiple geometries)
- Samples point clouds uniformly from mesh surfaces
- Transfers per-face labels to sampled points
- Normalizes to unit sphere (center to origin + scale)
- Saves as compressed `.npz` files with metadata

**Output:** `data/processed/` contains:
- One `.npz` file per sample with: points, semantic_labels, instance_labels, centroid, scale
- `preprocessing_stats.json` with dataset statistics

### 3. Train Model

Train PointNet for semantic part segmentation:

```bash
# Train with default settings (100 epochs, batch_size=8, lr=0.001)
python scripts/train.py

# Custom training configuration
python scripts/train.py --batch-size 16 --num-epochs 200 --lr 0.0005 --num-points 4096

# Resume from checkpoint
python scripts/train.py --resume models/checkpoints/best_model.pth

# View all training options
python scripts/train.py --help
```

**Training outputs:**
- Model checkpoints saved to `models/checkpoints/`
  - `best_model.pth` - Model with lowest validation loss
  - `final_model.pth` - Model after last epoch
  - `checkpoint_epoch_N.pth` - Periodic checkpoints (every 10 epochs by default)
  - `config.json` - Training configuration and hyperparameters
- TensorBoard logs in `runs/pointnet_<timestamp>/`

**Monitor training:**
```bash
tensorboard --logdir runs/
```

### 4. Evaluate Model

Evaluate trained model performance on the test set:

```bash
# Evaluate best model checkpoint
python scripts/evaluate.py

# Evaluate specific checkpoint
python scripts/evaluate.py --checkpoint models/checkpoints/final_model.pth

# Custom batch size for faster evaluation
python scripts/evaluate.py --batch-size 32
```

**Evaluation outputs:**
- Console output with overall accuracy, mean IoU, and per-class metrics
- Results saved to `results/evaluation_results_<timestamp>.json`
- Latest results in `results/latest_results.json`

**Metrics computed:**
- Overall point-wise accuracy
- Per-class accuracy (recall)
- Per-class IoU (Intersection over Union)
- Mean IoU (standard metric for segmentation)
- Confusion matrix for detailed analysis

### 5. Visualize Predictions

Generate interactive 3D visualizations of model predictions:

```bash
# Visualize 5 random test samples
python scripts/visualize_predictions.py

# Visualize specific samples
python scripts/visualize_predictions.py --sample-indices 0 5 10 15

# Show prediction errors (correct=green, incorrect=red)
python scripts/visualize_predictions.py --show-errors --num-samples 3

# Save without displaying (useful for servers)
python scripts/visualize_predictions.py --no-show --num-samples 10
```

**Visualization outputs:**
- Interactive HTML files in `results/visualizations/`
- Side-by-side ground truth vs predictions
- Color-coded point clouds by class
- Optional error visualization (correct/incorrect points)
- Per-sample accuracy in console

## Project Architecture

### Directory Structure

```
PartFlow3D/
├── configs/          # Model and training configuration files (YAML/JSON)
├── data/
│   ├── raw/         # Downloaded dataset (meshes + labels)
│   └── processed/   # Preprocessed data (point clouds, normalized meshes, etc.)
├── models/          # Saved model checkpoints and weights
├── notebooks/       # Jupyter notebooks for exploration and experimentation
├── scripts/         # Standalone utility scripts (download, preprocessing, etc.)
└── src/            # Core source code (models, training loops, data loaders)
```

### Model Architecture

**PointNet** (`src/models/pointnet.py`):
- Implements the PointNet architecture for semantic segmentation
- **TNet**: Spatial transformation network that learns input/feature alignment (3×3 and 64×64 transforms)
- **PointNetBackbone**: Feature extraction using shared MLPs and symmetric max pooling
- **PointNetSegmentation**: Combines per-point local features (64-dim) with global shape features (1024-dim) for segmentation
- Loss calculation includes feature transform regularization to encourage orthogonality
- Model is permutation-invariant (order of points doesn't matter)

**Key components:**
- Input transform: Aligns input point cloud to canonical orientation
- Feature transform: Aligns learned features in 64-dimensional space
- Max pooling: Creates global features from all points (achieves permutation invariance)
- Segmentation head: Concatenates local + global features → predicts per-point class labels

**Dataset handling** (`src/dataset.py`):
- `PartSegmentationDataset`: Loads preprocessed `.npz` files with point clouds and labels
- Automatic train/val/test splitting (70%/15%/15% by default)
- Data augmentation for training: random rotation (z-axis), jittering, scaling
- Point cloud subsampling/padding to fixed size
- Returns: points (N×3), semantic_labels (N,), instance_labels (N,)

### Development Flow

1. **Data Exploration** - Use `notebooks/01_data_exploration.ipynb` to visualize 3D data, understand segmentation labels, and analyze dataset statistics
2. **Preprocessing** - Run `scripts/preprocess_data.py` to convert meshes → point clouds with normalization
3. **Model Development** - Implement 3D segmentation architectures in `src/models/` (currently: PointNet)
4. **Training** - Use `scripts/train.py` with TensorBoard logging and checkpointing
5. **Evaluation** - Run `scripts/evaluate.py` to compute accuracy, IoU, and other metrics
6. **Visualization** - Use `scripts/visualize_predictions.py` to generate interactive 3D visualizations of predictions

### Data Exploration Notebook

`notebooks/01_data_exploration.ipynb` provides:
- Interactive 3D visualization (Plotly) of meshes and segmentations
- Dataset statistics (vertex counts, face counts, parts per mesh)
- Point cloud sampling demonstrations
- Understanding of semantic vs instance segmentation

### Key Technologies

- **3D Processing**: trimesh, open3d, PyMCubes for mesh/point cloud manipulation
- **Deep Learning**: PyTorch for model implementation and training
- **Visualization**: matplotlib, plotly, pyvista for 3D visualization
- **Data**: HuggingFace Hub for dataset downloads

## Code Quality

```bash
# Format code (Black formatter)
black .

# Lint code (Flake8)
flake8 .

# Run tests (when available)
pytest
```

## Important Implementation Details

### Dataset Label Format

**Critical:** Labels in PartObjaverse-Tiny are **per-face**, not per-vertex!
- Each label corresponds to a mesh face (triangle)
- When sampling point clouds, labels are assigned from the face each point was sampled from
- GLB files may contain Scene objects with multiple geometries - use `trimesh.util.concatenate()` to merge them
- **Number of classes**: Auto-detected by scanning all `.npz` files for max label value
  - Current dataset has 17 classes (labels 0-16)
  - Training script automatically determines this - no manual specification needed

### 3D Data Representations

The project explores multiple 3D data representations:
- **Meshes**: Vertices + faces (original format from dataset, GLB files)
- **Point Clouds**: Sampled points from mesh surfaces (used for neural networks, default: 2048 points)
- **Voxels**: 3D grid representation (memory-intensive but useful for certain architectures)

### Training Configuration

Default hyperparameters in `configs/pointnet_default.yaml`:
- **Data**: 2048 points per sample, 70/15/15 train/val/test split
- **Model**: PointNet with feature transform, dropout=0.3, reg_weight=0.001
- **Training**: batch_size=8, lr=0.001, Adam optimizer, 100 epochs
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)

All hyperparameters can be overridden via command-line arguments to `scripts/train.py`.

### Common Issues

**PyTorch version compatibility:**
- If you get errors about `verbose` parameter in `ReduceLROnPlateau`, you're using PyTorch >= 2.0
- The codebase is compatible with modern PyTorch versions (parameter has been removed)

**Label out of bounds errors:**
- The training script scans ALL preprocessed files to determine the true number of classes
- If you add new data, the number of classes will be auto-detected
- Semantic labels use indices 0 to N-1 for N classes

**Device warnings:**
- Pin memory warnings on MPS (Apple Silicon) can be ignored - they don't affect training
- Model defaults to CPU if CUDA is unavailable

## Next Steps for Development

**Completed:**
- ✓ Dataset download and preprocessing pipeline
- ✓ PointNet model implementation with feature transforms
- ✓ Training script with TensorBoard logging and checkpointing
- ✓ Data augmentation and dataset splitting
- ✓ Evaluation script with accuracy, IoU, and confusion matrix
- ✓ Interactive 3D visualization of predictions vs ground truth

**Potential next steps:**
1. **PointNet++**: Implement hierarchical feature learning for better local geometry understanding
2. **Instance segmentation**: Adapt model for instance-level predictions (currently semantic only)
3. **Hyperparameter tuning**: Experiment with different learning rates, architectures, point counts
4. **Advanced metrics**: Add precision, recall, F1-score per class
5. **Model comparison**: Compare PointNet vs PointNet++ vs other architectures
6. **Data exploration**: Analyze which object types/parts are hardest to segment

## Resources

- Dataset: [PartObjaverse-Tiny on HuggingFace](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny)
- Paper: [SAMPart3D](https://yhyang-myron.github.io/SAMPart3D-website/)
- PointNet Paper: [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
