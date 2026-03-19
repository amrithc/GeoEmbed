#!/usr/bin/env python
"""Stage 1: Download EuroSAT and flatten to data/tiles/ with metadata.csv.

Usage:
    python scripts/01_download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoembded.data.downloader import download_eurosat, flatten_to_tiles
from geoembded.config import METADATA_CSV, TILES_DIR


def main() -> None:
    eurosat_root = download_eurosat()
    flatten_to_tiles(source_dir=eurosat_root)

    # Verify
    import pandas as pd
    df = pd.read_csv(METADATA_CSV)
    tile_count = len(list(TILES_DIR.glob("*.jpg")))
    print(f"\n✓ {tile_count} tiles in {TILES_DIR}")
    print(f"✓ {len(df)} rows in metadata.csv")
    print(f"✓ Classes: {sorted(df['class_label'].unique())}")


if __name__ == "__main__":
    main()
