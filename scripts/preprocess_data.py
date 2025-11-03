#!/usr/bin/env python3
"""
Preprocess PartObjaverse-Tiny dataset for neural network training.

This script:
1. Loads 3D meshes from GLB files
2. Samples point clouds from mesh surfaces
3. Normalizes point clouds (center + unit sphere)
4. Saves preprocessed data with labels for efficient loading

Usage:
    python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed --num-points 2048
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """
    Load a mesh from GLB file, handling Scene vs Mesh objects.

    Args:
        mesh_path: Path to the GLB file

    Returns:
        trimesh.Trimesh object
    """
    loaded = trimesh.load(mesh_path)

    if isinstance(loaded, trimesh.Scene):
        # Concatenate all geometries in the scene
        # Labels in this dataset are per-face for the concatenated mesh
        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        return mesh
    else:
        return loaded


def sample_point_cloud(mesh: trimesh.Trimesh, labels: np.ndarray, num_points: int) -> tuple:
    """
    Sample a point cloud from a mesh surface with corresponding labels.

    Important: Labels in PartObjaverse-Tiny are PER-FACE, not per-vertex!

    Args:
        mesh: trimesh.Trimesh object
        labels: Per-face labels (semantic or instance)
        num_points: Number of points to sample

    Returns:
        points: (N, 3) array of point coordinates
        sampled_labels: (N,) array of labels for each point
    """
    # Verify that labels match mesh faces
    if len(labels) != len(mesh.faces):
        raise ValueError(
            f"Label count ({len(labels)}) does not match face count ({len(mesh.faces)}). "
            f"Labels should be per-face in this dataset."
        )

    # Sample points uniformly from mesh surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Assign label from the face that each point was sampled from
    sampled_labels = labels[face_indices]

    return points, sampled_labels


def normalize_point_cloud(points: np.ndarray) -> tuple:
    """
    Normalize a point cloud: center to origin and scale to unit sphere.

    Args:
        points: (N, 3) array of point coordinates

    Returns:
        normalized_points: (N, 3) array of normalized points
        centroid: (3,) array of original centroid
        scale: scalar, the scaling factor used
    """
    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Center to origin
    points_centered = points - centroid

    # Calculate max distance from origin (radius of bounding sphere)
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))

    # Scale to unit sphere
    scale = max_distance
    points_normalized = points_centered / scale

    return points_normalized, centroid, scale


def preprocess_sample(
    mesh_path: Path,
    semantic_label_path: Path,
    instance_label_path: Path,
    num_points: int,
    normalize: bool = True
) -> dict:
    """
    Preprocess a single sample: load mesh, sample point cloud, normalize.

    Args:
        mesh_path: Path to mesh file
        semantic_label_path: Path to semantic labels
        instance_label_path: Path to instance labels
        num_points: Number of points to sample
        normalize: Whether to normalize the point cloud

    Returns:
        Dictionary containing preprocessed data
    """
    # Load mesh
    mesh = load_mesh(mesh_path)

    # Load labels (per-face)
    semantic_labels = np.load(semantic_label_path)
    instance_labels = np.load(instance_label_path)

    # Sample point cloud
    points, semantic_point_labels = sample_point_cloud(mesh, semantic_labels, num_points)
    _, instance_point_labels = sample_point_cloud(mesh, instance_labels, num_points)

    # Normalize if requested
    if normalize:
        points, centroid, scale = normalize_point_cloud(points)
    else:
        centroid = np.zeros(3)
        scale = 1.0

    # Create output dictionary
    output = {
        'points': points.astype(np.float32),
        'semantic_labels': semantic_point_labels.astype(np.int32),
        'instance_labels': instance_point_labels.astype(np.int32),
        'centroid': centroid.astype(np.float32),
        'scale': np.float32(scale),
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'num_semantic_parts': len(np.unique(semantic_labels)),
        'num_instances': len(np.unique(instance_labels))
    }

    return output


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    num_points: int = 2048,
    normalize: bool = True
) -> None:
    """
    Preprocess the entire dataset.

    Args:
        input_dir: Directory containing raw data
        output_dir: Directory to save preprocessed data
        num_points: Number of points to sample per mesh
        normalize: Whether to normalize point clouds
    """
    # Define input directories
    mesh_dir = input_dir / "PartObjaverse-Tiny_mesh"
    semantic_gt_dir = input_dir / "PartObjaverse-Tiny_semantic_gt"
    instance_gt_dir = input_dir / "PartObjaverse-Tiny_instance_gt"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of mesh files
    mesh_files = sorted(list(mesh_dir.glob("*.glb")))

    logger.info(f"Preprocessing {len(mesh_files)} samples...")
    logger.info(f"Number of points per sample: {num_points}")
    logger.info(f"Normalization: {'enabled' if normalize else 'disabled'}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Track statistics
    stats = {
        'num_samples': len(mesh_files),
        'num_points': num_points,
        'normalized': normalize,
        'samples': []
    }

    # Process each sample
    failed_samples = []

    for mesh_path in tqdm(mesh_files, desc="Processing samples"):
        sample_id = mesh_path.stem

        try:
            # Define label paths
            semantic_label_path = semantic_gt_dir / f"{sample_id}.npy"
            instance_label_path = instance_gt_dir / f"{sample_id}.npy"

            # Preprocess sample
            preprocessed = preprocess_sample(
                mesh_path,
                semantic_label_path,
                instance_label_path,
                num_points,
                normalize
            )

            # Save preprocessed data
            output_path = output_dir / f"{sample_id}.npz"
            np.savez_compressed(
                output_path,
                points=preprocessed['points'],
                semantic_labels=preprocessed['semantic_labels'],
                instance_labels=preprocessed['instance_labels'],
                centroid=preprocessed['centroid'],
                scale=preprocessed['scale']
            )

            # Track sample statistics
            stats['samples'].append({
                'id': sample_id,
                'num_vertices': int(preprocessed['num_vertices']),
                'num_faces': int(preprocessed['num_faces']),
                'num_semantic_parts': int(preprocessed['num_semantic_parts']),
                'num_instances': int(preprocessed['num_instances'])
            })

        except Exception as e:
            logger.error(f"Failed to process {sample_id}: {e}")
            failed_samples.append(sample_id)
            continue

    # Save statistics
    stats_path = output_dir / "preprocessing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Successfully processed: {len(stats['samples'])}/{len(mesh_files)} samples")

    if failed_samples:
        logger.warning(f"Failed samples ({len(failed_samples)}): {failed_samples[:10]}...")

    logger.info(f"Preprocessed data saved to: {output_dir.absolute()}")
    logger.info(f"Statistics saved to: {stats_path.absolute()}")

    # Print summary statistics
    if stats['samples']:
        num_semantic_parts = [s['num_semantic_parts'] for s in stats['samples']]
        num_instances = [s['num_instances'] for s in stats['samples']]

        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Semantic parts per sample: {np.mean(num_semantic_parts):.1f} ± {np.std(num_semantic_parts):.1f}")
        logger.info(f"  Instances per sample: {np.mean(num_instances):.1f} ± {np.std(num_instances):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PartObjaverse-Tiny dataset for neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess with default settings (2048 points, normalized)
  python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed

  # Use more points for higher resolution
  python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed --num-points 4096

  # Disable normalization
  python scripts/preprocess_data.py --input-dir data/raw --output-dir data/processed --no-normalize
        """
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw data (default: data/raw)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save preprocessed data (default: data/processed)"
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points to sample per mesh (default: 2048)"
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable point cloud normalization"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run preprocessing
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()
