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

### Development Flow

1. **Data Exploration** - Use `notebooks/01_data_exploration.ipynb` to visualize 3D data, understand segmentation labels, and analyze dataset statistics
2. **Preprocessing** - Run `scripts/preprocess_data.py` to convert meshes → point clouds with normalization
3. **Model Development** - Implement 3D segmentation architectures (PointNet, PointNet++, etc.) in `src/`
4. **Training** - Create training scripts with proper logging (TensorBoard), checkpointing
5. **Evaluation** - Analyze segmentation performance and understand trade-offs

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

### 3D Data Representations

The project explores multiple 3D data representations:
- **Meshes**: Vertices + faces (original format from dataset, GLB files)
- **Point Clouds**: Sampled points from mesh surfaces (used for neural networks, default: 2048 points)
- **Voxels**: 3D grid representation (memory-intensive but useful for certain architectures)

### Environment Variables

Key environment variables in `.env` (see `.env.example`):
- `DATA_DIR`, `PROCESSED_DATA_DIR` - Data paths
- `MODEL_DIR`, `CHECKPOINT_DIR` - Model storage
- `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE` - Training hyperparameters
- `TENSORBOARD_LOG_DIR` - Logging directory for TensorBoard

### TensorBoard Monitoring

```bash
# View training metrics (after training starts)
tensorboard --logdir runs/
```

## Resources

- Dataset: [PartObjaverse-Tiny on HuggingFace](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny)
- Paper: [SAMPart3D](https://yhyang-myron.github.io/SAMPart3D-website/)
