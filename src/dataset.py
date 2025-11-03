"""
PyTorch Dataset for PartObjaverse-Tiny point cloud data.

This module provides a Dataset class for loading preprocessed point clouds
with segmentation labels, along with data augmentation utilities.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging


logger = logging.getLogger(__name__)


class PartSegmentationDataset(Dataset):
    """
    Dataset for 3D part segmentation using point clouds.

    Loads preprocessed point clouds from .npz files and applies optional
    data augmentation (rotation, jittering, scaling).

    Args:
        data_dir: Directory containing preprocessed .npz files
        split: 'train', 'val', or 'test'
        train_ratio: Fraction of data to use for training (default: 0.7)
        val_ratio: Fraction of data to use for validation (default: 0.15)
        augment: Whether to apply data augmentation (default: True for train)
        num_points: Number of points to use (default: None = use all)
        normalize: Whether points are already normalized (default: True)
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        augment: Optional[bool] = None,
        num_points: Optional[int] = None,
        normalize: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment if augment is not None else (split == 'train')
        self.num_points = num_points
        self.normalize = normalize

        # Get all .npz files
        self.files = sorted(list(self.data_dir.glob("*.npz")))

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Split into train/val/test
        total = len(self.files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        if split == 'train':
            self.files = self.files[:train_size]
        elif split == 'val':
            self.files = self.files[train_size:train_size + val_size]
        elif split == 'test':
            self.files = self.files[train_size + val_size:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        logger.info(f"Loaded {len(self.files)} samples for {split} split")

        # Track label statistics
        self._analyze_labels()

    def _analyze_labels(self) -> None:
        """Analyze label distribution across the dataset."""
        all_semantic_labels = set()
        all_instance_labels = set()

        # Sample a few files to get label statistics
        sample_size = min(10, len(self.files))
        for i in range(sample_size):
            data = np.load(self.files[i])
            all_semantic_labels.update(data['semantic_labels'])
            all_instance_labels.update(data['instance_labels'])

        self.num_semantic_classes = len(all_semantic_labels)
        self.num_instance_classes = len(all_instance_labels)

        logger.info(f"Approximate semantic classes: {self.num_semantic_classes}")
        logger.info(f"Approximate instance classes: {self.num_instance_classes}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.

        Returns:
            Dictionary containing:
                - points: (N, 3) point coordinates
                - semantic_labels: (N,) semantic segmentation labels
                - instance_labels: (N,) instance segmentation labels
                - sample_id: string identifier
        """
        # Load data
        file_path = self.files[idx]
        data = np.load(file_path)

        points = data['points'].astype(np.float32)  # (N, 3)
        semantic_labels = data['semantic_labels'].astype(np.int64)  # (N,)
        instance_labels = data['instance_labels'].astype(np.int64)  # (N,)

        # Subsample if requested
        if self.num_points is not None and len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            semantic_labels = semantic_labels[indices]
            instance_labels = instance_labels[indices]
        elif self.num_points is not None and len(points) < self.num_points:
            # Pad if not enough points
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
            semantic_labels = semantic_labels[indices]
            instance_labels = instance_labels[indices]

        # Apply augmentation
        if self.augment:
            points = self._augment_point_cloud(points)

        # Convert to torch tensors
        sample = {
            'points': torch.from_numpy(points),
            'semantic_labels': torch.from_numpy(semantic_labels),
            'instance_labels': torch.from_numpy(instance_labels),
            'sample_id': file_path.stem
        }

        return sample

    def _augment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to point cloud.

        Augmentations:
        - Random rotation around up-axis (z-axis)
        - Random jittering (add small noise)
        - Random scaling

        Args:
            points: (N, 3) array of point coordinates

        Returns:
            Augmented points
        """
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T

        # Random jittering (Gaussian noise)
        jitter_std = 0.01
        jitter = np.random.normal(0, jitter_std, points.shape)
        points = points + jitter

        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        return points.astype(np.float32)


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_points: Optional[int] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        num_points: Number of points per sample (None = use all)
        **kwargs: Additional arguments passed to Dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = PartSegmentationDataset(
        data_dir,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augment=True,
        num_points=num_points,
        **kwargs
    )

    val_dataset = PartSegmentationDataset(
        data_dir,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augment=False,
        num_points=num_points,
        **kwargs
    )

    test_dataset = PartSegmentationDataset(
        data_dir,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augment=False,
        num_points=num_points,
        **kwargs
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data/processed")

    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        print("Run preprocessing first: python scripts/preprocess_data.py")
        exit(1)

    # Create a dataset
    dataset = PartSegmentationDataset(data_dir, split='train')

    print(f"\nDataset size: {len(dataset)}")
    print(f"Semantic classes: {dataset.num_semantic_classes}")
    print(f"Instance classes: {dataset.num_instance_classes}")

    # Load a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Points shape: {sample['points'].shape}")
    print(f"  Semantic labels shape: {sample['semantic_labels'].shape}")
    print(f"  Instance labels shape: {sample['instance_labels'].shape}")
    print(f"  Sample ID: {sample['sample_id']}")

    # Test dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=4,
        num_workers=0  # Use 0 for testing
    )

    # Load a batch
    batch = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  Points shape: {batch['points'].shape}")
    print(f"  Semantic labels shape: {batch['semantic_labels'].shape}")
    print(f"  Instance labels shape: {batch['instance_labels'].shape}")
    print(f"  Sample IDs: {batch['sample_id'][:2]}...")
