"""
Evaluation script for PointNet segmentation model.

Loads a trained model checkpoint and evaluates on the test set, computing:
- Overall accuracy
- Per-class accuracy
- Per-class IoU (Intersection over Union)
- Mean IoU
- Confusion matrix

Results are saved to JSON and printed to console.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import logging
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from src.models.pointnet import PointNetSegmentation
from src.dataset import create_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    """Evaluates semantic segmentation performance."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Confusion matrix: rows=ground truth, cols=predictions
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_correct = 0
        self.total_points = 0

    def update(self, predictions: np.ndarray, labels: np.ndarray):
        """
        Update metrics with a batch of predictions.

        Args:
            predictions: (N,) predicted class indices
            labels: (N,) ground truth class indices
        """
        predictions = predictions.flatten()
        labels = labels.flatten()

        # Update confusion matrix
        for pred, label in zip(predictions, labels):
            if 0 <= label < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[label, pred] += 1

        # Update accuracy
        correct = (predictions == labels).sum()
        self.total_correct += correct
        self.total_points += len(labels)

    def get_metrics(self) -> dict:
        """
        Compute all metrics from confusion matrix.

        Returns:
            Dictionary containing:
                - overall_accuracy: Overall point-wise accuracy
                - per_class_accuracy: Accuracy for each class
                - per_class_iou: IoU for each class
                - mean_iou: Mean IoU across all classes
                - confusion_matrix: Raw confusion matrix
        """
        # Overall accuracy
        overall_accuracy = self.total_correct / max(self.total_points, 1)

        # Per-class accuracy (recall)
        per_class_accuracy = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            total_class = self.confusion_matrix[i, :].sum()
            if total_class > 0:
                per_class_accuracy[i] = self.confusion_matrix[i, i] / total_class

        # Per-class IoU
        per_class_iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            # True positives
            tp = self.confusion_matrix[i, i]
            # False positives (predicted as class i but actually other classes)
            fp = self.confusion_matrix[:, i].sum() - tp
            # False negatives (actually class i but predicted as other classes)
            fn = self.confusion_matrix[i, :].sum() - tp

            union = tp + fp + fn
            if union > 0:
                per_class_iou[i] = tp / union

        # Mean IoU (only over classes that appear in ground truth)
        classes_present = self.confusion_matrix.sum(axis=1) > 0
        mean_iou = per_class_iou[classes_present].mean() if classes_present.any() else 0.0

        metrics = {
            'overall_accuracy': float(overall_accuracy),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'per_class_iou': per_class_iou.tolist(),
            'mean_iou': float(mean_iou),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'num_classes': self.num_classes,
            'total_points': int(self.total_points)
        }

        return metrics

    def print_metrics(self, metrics: dict):
        """Print metrics in a readable format."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)

        print(f"\nTotal points evaluated: {metrics['total_points']:,}")
        print(f"Overall accuracy: {metrics['overall_accuracy']*100:.2f}%")
        print(f"Mean IoU: {metrics['mean_iou']*100:.2f}%")

        print("\nPer-class metrics:")
        print(f"{'Class':<10} {'Accuracy':<12} {'IoU':<12} {'Support':<12}")
        print("-" * 50)

        confusion_matrix = np.array(metrics['confusion_matrix'])
        for i in range(metrics['num_classes']):
            support = confusion_matrix[i, :].sum()
            if support > 0:
                acc = metrics['per_class_accuracy'][i] * 100
                iou = metrics['per_class_iou'][i] * 100
                print(f"{i:<10} {acc:>10.2f}% {iou:>10.2f}% {support:>10,}")

        print("="*60 + "\n")


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str,
    num_classes: int
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained PointNet model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        num_classes: Number of segmentation classes

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    evaluator = SegmentationEvaluator(num_classes)

    logger.info(f"Evaluating on {len(test_loader.dataset)} test samples...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            points = batch['points'].to(device)  # (B, N, 3)
            labels = batch['semantic_labels'].to(device)  # (B, N)

            # Forward pass
            logits, _ = model(points)  # (B, num_classes, N)

            # Get predictions
            predictions = torch.argmax(logits, dim=1)  # (B, N)

            # Update metrics
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            evaluator.update(predictions_np, labels_np)

    # Compute final metrics
    metrics = evaluator.get_metrics()
    evaluator.print_metrics(metrics)

    return metrics


def load_checkpoint(checkpoint_path: Path, num_classes: int, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = PointNetSegmentation(
        num_classes=num_classes,
        use_feature_transform=True,
        dropout=0.3
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Print checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        logger.info(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate PointNet segmentation model')

    # Model
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')

    # Data
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to preprocessed data')
    parser.add_argument('--num-points', type=int, default=2048,
                        help='Number of points per sample')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Train a model first: python scripts/train.py")
        return

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run preprocessing first: python scripts/preprocess_data.py")
        return

    # Auto-detect number of classes
    if args.num_classes is None:
        from scripts.train import get_num_classes
        args.num_classes = get_num_classes(data_dir)

    # Create test dataloader
    logger.info("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_points=args.num_points
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(checkpoint_path, args.num_classes, device)

    # Evaluate
    metrics = evaluate_model(model, test_loader, device, args.num_classes)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'evaluation_results_{timestamp}.json'

    results = {
        'checkpoint': str(checkpoint_path),
        'data_dir': str(data_dir),
        'num_points': args.num_points,
        'batch_size': args.batch_size,
        'device': str(device),
        'timestamp': timestamp,
        'metrics': metrics
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Also save a summary
    summary_file = output_dir / 'latest_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Latest results also saved to {summary_file}")


if __name__ == "__main__":
    main()
