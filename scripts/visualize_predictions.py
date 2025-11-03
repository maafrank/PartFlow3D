"""
Visualization script for PointNet segmentation predictions.

Loads a trained model and visualizes predictions on test samples:
- Side-by-side comparison of ground truth vs predictions
- Color-coded point clouds by class
- Interactive 3D visualizations (Plotly)
- Optional: Save static images or HTML files
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.pointnet import PointNetSegmentation
from src.dataset import PartSegmentationDataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Color palette for segmentation classes (17 distinct colors)
COLOR_PALETTE = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]


def get_color_for_label(label: int) -> str:
    """Get color for a class label."""
    return COLOR_PALETTE[label % len(COLOR_PALETTE)]


def visualize_point_cloud_comparison(
    points: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    sample_id: str,
    save_path: Path = None,
    show: bool = True
):
    """
    Create side-by-side visualization of ground truth vs predictions.

    Args:
        points: (N, 3) point cloud coordinates
        ground_truth: (N,) ground truth labels
        predictions: (N,) predicted labels
        sample_id: Sample identifier for title
        save_path: Optional path to save HTML file
        show: Whether to display the plot
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Ground Truth', 'Predictions'),
        horizontal_spacing=0.05
    )

    # Ground truth visualization
    gt_colors = [get_color_for_label(int(label)) for label in ground_truth]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=gt_colors,
            ),
            text=[f'Class: {int(label)}' for label in ground_truth],
            hovertemplate='<b>Class %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # Predictions visualization
    pred_colors = [get_color_for_label(int(label)) for label in predictions]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=pred_colors,
            ),
            text=[f'Class: {int(label)}' for label in predictions],
            hovertemplate='<b>Class %{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=f'Segmentation Results - {sample_id}',
        height=600,
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        scene2=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        )
    )

    # Save if requested
    if save_path:
        fig.write_html(str(save_path))
        logger.info(f"Saved visualization to {save_path}")

    # Show if requested
    if show:
        fig.show()

    return fig


def visualize_errors(
    points: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    sample_id: str,
    save_path: Path = None,
    show: bool = True
):
    """
    Visualize prediction errors.

    Args:
        points: (N, 3) point cloud coordinates
        ground_truth: (N,) ground truth labels
        predictions: (N,) predicted labels
        sample_id: Sample identifier for title
        save_path: Optional path to save HTML file
        show: Whether to display the plot
    """
    # Calculate errors
    errors = (ground_truth != predictions).astype(int)
    error_rate = errors.mean()

    # Color: green for correct, red for incorrect
    colors = ['#2ecc71' if e == 0 else '#e74c3c' for e in errors]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
            ),
            text=[f'GT: {int(gt)}, Pred: {int(pred)}, {"✓" if e == 0 else "✗"}'
                  for gt, pred, e in zip(ground_truth, predictions, errors)],
            hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        )
    ])

    fig.update_layout(
        title=f'Prediction Errors - {sample_id}<br>Error Rate: {error_rate*100:.2f}%',
        height=600,
        scene=dict(
            aspectmode='data',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        )
    )

    if save_path:
        fig.write_html(str(save_path))
        logger.info(f"Saved error visualization to {save_path}")

    if show:
        fig.show()

    return fig


def load_checkpoint(checkpoint_path: Path, num_classes: int, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PointNetSegmentation(
        num_classes=num_classes,
        use_feature_transform=True,
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize PointNet segmentation predictions')

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

    # Visualization
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--sample-indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize (e.g., 0 5 10)')
    parser.add_argument('--show-errors', action='store_true',
                        help='Also visualize prediction errors')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save)')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                        help='Directory to save visualizations')

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

    # Create test dataset
    logger.info("Loading test dataset...")
    test_dataset = PartSegmentationDataset(
        data_dir,
        split='test',
        augment=False,
        num_points=args.num_points
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_checkpoint(checkpoint_path, args.num_classes, device)

    # Determine which samples to visualize
    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        # Random samples
        sample_indices = np.random.choice(
            len(test_dataset),
            min(args.num_samples, len(test_dataset)),
            replace=False
        ).tolist()

    logger.info(f"Visualizing {len(sample_indices)} samples: {sample_indices}")

    # Visualize samples
    show = not args.no_show

    for idx in tqdm(sample_indices, desc="Generating visualizations"):
        # Load sample
        sample = test_dataset[idx]
        points = sample['points'].numpy()  # (N, 3)
        ground_truth = sample['semantic_labels'].numpy()  # (N,)
        sample_id = sample['sample_id']

        # Run inference
        with torch.no_grad():
            points_tensor = sample['points'].unsqueeze(0).to(device)  # (1, N, 3)
            logits, _ = model(points_tensor)  # (1, num_classes, N)
            predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (N,)

        # Visualize comparison
        comparison_path = output_dir / f'{sample_id}_comparison.html'
        visualize_point_cloud_comparison(
            points,
            ground_truth,
            predictions,
            sample_id,
            save_path=comparison_path,
            show=show
        )

        # Visualize errors if requested
        if args.show_errors:
            error_path = output_dir / f'{sample_id}_errors.html'
            visualize_errors(
                points,
                ground_truth,
                predictions,
                sample_id,
                save_path=error_path,
                show=show
            )

        # Calculate sample accuracy
        accuracy = (predictions == ground_truth).mean()
        logger.info(f"Sample {sample_id}: Accuracy = {accuracy*100:.2f}%")

    logger.info(f"\nAll visualizations saved to {output_dir}")
    logger.info(f"Open the HTML files in a browser to view interactive 3D visualizations")


if __name__ == "__main__":
    main()
