"""
Training script for PointNet segmentation model.

Trains on PartObjaverse-Tiny preprocessed point clouds with:
- TensorBoard logging for metrics visualization
- Model checkpointing (best validation loss + periodic saves)
- Learning rate scheduling
- Validation evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import json

from src.models.pointnet import PointNetSegmentation
from src.dataset import create_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Handles training and validation loops for PointNet segmentation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: str,
        checkpoint_dir: Path,
        log_dir: Path,
        reg_weight: float = 0.001
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.reg_weight = reg_weight

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

        # Tracking
        self.best_val_loss = float('inf')
        self.current_epoch = 0

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        logger.info(f"TensorBoard log dir: {log_dir}")

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            points = batch['points'].to(self.device)  # (B, N, 3)
            labels = batch['semantic_labels'].to(self.device)  # (B, N)

            # Forward pass
            self.optimizer.zero_grad()
            logits, feature_trans = self.model(points)

            # Calculate loss
            loss = self.model.get_loss(
                logits,
                labels,
                feature_trans,
                reg_weight=self.reg_weight
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            predictions = torch.argmax(logits, dim=1)  # (B, N)
            correct = (predictions == labels).sum().item()
            num_points = labels.numel()

            total_loss += loss.item()
            total_correct += correct
            total_points += num_points

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct / num_points:.2f}%"
            })

            # Log to TensorBoard (every 10 batches)
            if batch_idx % 10 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar(
                    'train/batch_accuracy',
                    100.0 * correct / num_points,
                    global_step
                )

        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = 100.0 * total_correct / total_points

        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }

        return metrics

    def validate_epoch(self) -> dict:
        """Run validation epoch."""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for batch in pbar:
                # Move data to device
                points = batch['points'].to(self.device)
                labels = batch['semantic_labels'].to(self.device)

                # Forward pass
                logits, feature_trans = self.model(points)

                # Calculate loss
                loss = self.model.get_loss(
                    logits,
                    labels,
                    feature_trans,
                    reg_weight=self.reg_weight
                )

                # Metrics
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == labels).sum().item()
                num_points = labels.numel()

                total_loss += loss.item()
                total_correct += correct
                total_points += num_points

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.0 * correct / num_points:.2f}%"
                })

        # Epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = 100.0 * total_correct / total_points

        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }

        return metrics

    def train(self, num_epochs: int, save_freq: int = 10):
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])

            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth', val_metrics)
                logger.info(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', val_metrics)
                logger.info(f"✓ Saved checkpoint at epoch {epoch + 1}")

        # Save final model
        self.save_checkpoint('final_model.pth', val_metrics)
        logger.info("Training complete!")
        self.writer.close()

    def save_checkpoint(self, filename: str, metrics: dict):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, checkpoint_path)


def get_num_classes(data_dir: Path) -> int:
    """Determine number of classes by scanning dataset."""
    import numpy as np

    npz_files = list(data_dir.glob("*.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")

    # Scan ALL files to determine max label
    max_label = 0
    logger.info(f"Scanning {len(npz_files)} files to determine number of classes...")

    for npz_file in npz_files:
        data = np.load(npz_file)
        max_label = max(max_label, data['semantic_labels'].max())

    num_classes = int(max_label) + 1
    logger.info(f"Detected {num_classes} classes from dataset (max label: {max_label})")

    return num_classes


def main():
    parser = argparse.ArgumentParser(description='Train PointNet for part segmentation')

    # Data
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Path to preprocessed data')
    parser.add_argument('--num-points', type=int, default=2048,
                        help='Number of points per sample')

    # Model
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--use-feature-transform', action='store_true', default=True,
                        help='Use feature transformation')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--reg-weight', type=float, default=0.001,
                        help='Feature transform regularization weight')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='TensorBoard log directory')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"pointnet_{timestamp}"

    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        logger.error("Run preprocessing first: python scripts/preprocess_data.py")
        return

    # Auto-detect number of classes
    if args.num_classes is None:
        args.num_classes = get_num_classes(data_dir)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_points=args.num_points
    )

    # Create model
    logger.info("Creating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointNetSegmentation(
        num_classes=args.num_classes,
        use_feature_transform=args.use_feature_transform,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    # Save training configuration
    config = vars(args)
    config['device'] = str(device)
    config['total_params'] = total_params
    config['train_samples'] = len(train_loader.dataset)
    config['val_samples'] = len(val_loader.dataset)

    config_path = checkpoint_dir / 'config.json'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_path}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        reg_weight=args.reg_weight
    )

    # Train
    trainer.train(num_epochs=args.num_epochs, save_freq=args.save_freq)

    logger.info(f"\nTo view training metrics, run:")
    logger.info(f"  tensorboard --logdir {log_dir.parent}")


if __name__ == "__main__":
    main()
