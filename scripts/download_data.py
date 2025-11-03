#!/usr/bin/env python3
"""
Download PartObjaverse-Tiny dataset from HuggingFace.

This script downloads the PartObjaverse-Tiny dataset which contains:
- 3D mesh data (.obj files)
- Part segmentation labels
- Point cloud representations

Usage:
    python scripts/download_data.py --output-dir data/raw --subset train
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset(
    output_dir: Path,
    dataset_name: str = "yhyang-myron/PartObjaverse-Tiny",
    subset: Optional[str] = None,
    cache_dir: Optional[Path] = None
) -> None:
    """
    Download the PartObjaverse-Tiny dataset from HuggingFace.

    Args:
        output_dir: Directory to save the downloaded data
        dataset_name: HuggingFace dataset identifier
        subset: Optional subset to download (train/test/validation)
        cache_dir: Optional cache directory for HuggingFace downloads
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {cache_dir.absolute()}")

    try:
        # Download using HuggingFace datasets library
        logger.info("Starting download...")

        if subset:
            logger.info(f"Downloading subset: {subset}")
            dataset = load_dataset(
                dataset_name,
                split=subset,
                cache_dir=str(cache_dir) if cache_dir else None
            )
        else:
            logger.info("Downloading all splits")
            dataset = load_dataset(
                dataset_name,
                cache_dir=str(cache_dir) if cache_dir else None
            )

        # Save the dataset to disk
        logger.info(f"Saving dataset to {output_dir}")
        dataset.save_to_disk(str(output_dir))

        logger.info("Download completed successfully!")
        logger.info(f"Dataset saved to: {output_dir.absolute()}")

        # Print dataset info
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                logger.info(f"Split '{split_name}': {len(split_data)} samples")
        else:
            logger.info(f"Total samples: {len(dataset)}")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.error("Make sure you have internet connection and required packages installed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download PartObjaverse-Tiny dataset for 3D segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire dataset
  python scripts/download_data.py --output-dir data/raw

  # Download only training set
  python scripts/download_data.py --output-dir data/raw --subset train

  # Specify custom cache directory
  python scripts/download_data.py --output-dir data/raw --cache-dir .cache
        """
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded data (default: data/raw)"
    )

    parser.add_argument(
        "--subset",
        type=str,
        choices=["train", "test", "validation"],
        help="Download specific subset only (default: download all)"
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache directory for HuggingFace downloads (default: .cache)"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yhyang-myron/PartObjaverse-Tiny",
        help="HuggingFace dataset name (default: yhyang-myron/PartObjaverse-Tiny)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    download_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        subset=args.subset,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
