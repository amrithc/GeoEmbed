#!/usr/bin/env python
"""Stage 0 (optional): Prepare a custom image dataset for GeoEmbed.

Use this instead of 01_download_data.py when you have your own images.

Expected input layout — one subdirectory per class label:
    my_images/
        Forest/
            img001.jpg
            img002.jpg
        Urban/
            img003.tif
        ...

If your images have no class structure, put them all in one subfolder:
    my_images/
        unlabeled/
            img001.jpg
            img002.jpg

The subdirectory name becomes the class_label in metadata.csv.

Usage:
    python scripts/00_prepare_custom_data.py --source-dir /path/to/my_images
    python scripts/00_prepare_custom_data.py --source-dir /path/to/my_images \\
        --tiles-dir data/tiles --metadata-csv data/metadata.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoembded.config import TILES_DIR, METADATA_CSV
from geoembded.data.downloader import flatten_to_tiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a custom image dataset for GeoEmbed."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Root directory containing subfolders of images (one folder per class label).",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=TILES_DIR,
        help=f"Output flat tile directory (default: {TILES_DIR})",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV,
        help=f"Output metadata CSV (default: {METADATA_CSV})",
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Error: --source-dir '{args.source_dir}' does not exist.")
        sys.exit(1)

    print(f"Source:   {args.source_dir}")
    print(f"Tiles:    {args.tiles_dir}")
    print(f"Metadata: {args.metadata_csv}\n")

    flatten_to_tiles(
        source_dir=args.source_dir,
        output_dir=args.tiles_dir,
        metadata_csv=args.metadata_csv,
    )

    # Show class distribution
    import pandas as pd
    df = pd.read_csv(args.metadata_csv)
    print("\nClass distribution:")
    print(df["class_label"].value_counts().to_string())
    print(f"\nTotal: {len(df)} tiles")
    print(f"\nNext: python scripts/02_build_embeddings.py --model dinov2")


if __name__ == "__main__":
    main()
