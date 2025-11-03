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

## Next Steps

1. **Data Exploration**: Explore the 3D data format, visualization, and segmentation labels
2. **Preprocessing**: Convert meshes to point clouds, apply normalization
3. **Model Development**: Implement 3D segmentation networks (PointNet, PointNet++, etc.)
4. **Training**: Train and evaluate segmentation models
5. **Analysis**: Understand trade-offs in 3D data representation and model architectures

## Goals

- Understand 3D data formats (meshes, point clouds, voxels)
- Learn preprocessing pipelines for 3D data
- Experiment with neural networks for 3D segmentation
- Explore trade-offs in model architecture, data representation, and performance
- Gain practical experience relevant to dental design workflows at companies like cadflow.ai