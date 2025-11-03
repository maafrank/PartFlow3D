#!/usr/bin/env python3
"""
Download PartObjaverse-Tiny dataset from HuggingFace.

This script downloads the PartObjaverse-Tiny dataset which contains:
- 3D mesh data (.obj files)
- Part segmentation labels (semantic and instance)

Usage:
    python scripts/download_data.py --output-dir data/raw
"""

import argparse
import logging
from pathlib import Path
import sys
import zipfile

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
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

# Files to download from the dataset
DATASET_FILES = [
    "PartObjaverse-Tiny_mesh.zip",
    "PartObjaverse-Tiny_semantic_gt.zip",
    "PartObjaverse-Tiny_instance_gt.zip"
]


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file to a specified directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
    """
    logger.info(f"Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()

        # Extract with progress bar
        for file in tqdm(file_list, desc=f"Extracting {zip_path.name}", unit="files"):
            zip_ref.extract(file, extract_to)

    logger.info(f"Extracted to {extract_to}")


def download_dataset(
    output_dir: Path,
    dataset_name: str = "yhyang-myron/PartObjaverse-Tiny",
    extract: bool = True,
    keep_zip: bool = False
) -> None:
    """
    Download the PartObjaverse-Tiny dataset from HuggingFace.

    Args:
        output_dir: Directory to save the downloaded data
        dataset_name: HuggingFace dataset identifier
        extract: Whether to extract zip files after downloading
        keep_zip: Whether to keep zip files after extraction
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = output_dir / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Files to download: {len(DATASET_FILES)}")

    downloaded_files = []

    try:
        # Download each file
        for filename in DATASET_FILES:
            logger.info(f"\nDownloading {filename}...")

            file_path = hf_hub_download(
                repo_id=dataset_name,
                filename=filename,
                repo_type="dataset",
                local_dir=str(zip_dir),
                local_dir_use_symlinks=False
            )

            downloaded_files.append(Path(file_path))
            logger.info(f"Downloaded: {file_path}")

        logger.info("\n" + "="*60)
        logger.info("All files downloaded successfully!")
        logger.info("="*60)

        # Extract if requested
        if extract:
            logger.info("\nExtracting files...")
            for zip_path in downloaded_files:
                extract_zip(zip_path, output_dir)

            # Remove zip files if not keeping them
            if not keep_zip:
                logger.info("\nCleaning up zip files...")
                for zip_path in downloaded_files:
                    zip_path.unlink()
                    logger.info(f"Removed {zip_path.name}")
                # Remove zip directory if empty
                if zip_dir.exists() and not any(zip_dir.iterdir()):
                    zip_dir.rmdir()
                    logger.info(f"Removed empty directory {zip_dir}")

        logger.info("\n" + "="*60)
        logger.info("Dataset download complete!")
        logger.info(f"Location: {output_dir.absolute()}")
        logger.info("="*60)

        # Print directory structure
        logger.info("\nDataset structure:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_dir():
                rel_path = item.relative_to(output_dir)
                logger.info(f"  📁 {rel_path}/")
                # Count files in directory
                file_count = len([f for f in item.iterdir() if f.is_file()])
                if file_count > 0:
                    logger.info(f"     ({file_count} files)")

    except Exception as e:
        logger.error(f"\nError downloading dataset: {e}")
        logger.error("Make sure you have internet connection and required packages installed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download PartObjaverse-Tiny dataset for 3D segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and extract dataset
  python scripts/download_data.py --output-dir data/raw

  # Download but don't extract
  python scripts/download_data.py --output-dir data/raw --no-extract

  # Keep zip files after extraction
  python scripts/download_data.py --output-dir data/raw --keep-zip
        """
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded data (default: data/raw)"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yhyang-myron/PartObjaverse-Tiny",
        help="HuggingFace dataset name (default: yhyang-myron/PartObjaverse-Tiny)"
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract zip files after downloading"
    )

    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip files after extraction (default: remove them)"
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
        extract=not args.no_extract,
        keep_zip=args.keep_zip
    )


if __name__ == "__main__":
    main()
