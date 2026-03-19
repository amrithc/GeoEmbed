#!/usr/bin/env python
"""Stage 2: Generate embeddings for all tiles and store in Zarr.

Usage:
    python scripts/02_build_embeddings.py --model resnet50
    python scripts/02_build_embeddings.py --model dinov2
    python scripts/02_build_embeddings.py --model clip
    python scripts/02_build_embeddings.py --model all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from geoembded.config import MODEL_CONFIGS, METADATA_CSV, TILES_DIR
from geoembded.data.tile_dataset import TileDataset, default_transform, clip_transform
from geoembded.storage.zarr_store import create_zarr_store, write_embeddings, zarr_info


def build_for_model(model_name: str, tiles_dir: Path, metadata_csv: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Building embeddings: {model_name}")
    print(f"{'='*60}")

    cfg = MODEL_CONFIGS[model_name]

    # Choose the right transform for CLIP vs everything else
    transform = clip_transform(cfg["input_size"]) if model_name == "clip" else default_transform(cfg["input_size"])

    dataset = TileDataset(tiles_dir=tiles_dir, metadata_csv=metadata_csv, transform=transform)
    n = len(dataset)
    print(f"Dataset: {n} tiles")

    # Create Zarr store
    create_zarr_store(model_name, n_samples=n, embed_dim=cfg["embed_dim"])

    # Load the embedder
    if model_name == "resnet50":
        from geoembded.embeddings.resnet import ResNet50Embedder
        embedder = ResNet50Embedder()
    elif model_name == "dinov2":
        from geoembded.embeddings.dinov2 import DINOv2Embedder
        embedder = DINOv2Embedder()
    elif model_name == "clip":
        from geoembded.embeddings.clip_model import CLIPEmbedder
        embedder = CLIPEmbedder()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    embeddings, tile_ids, class_labels = embedder.embed_dataset(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=0,  # 0 is safest for MPS
    )

    # Write embeddings to Zarr
    write_embeddings(model_name, embeddings)

    # Write tile_ids + class_labels to a per-model CSV (join key = row index)
    meta_out = Path(f"embeddings/{model_name}/metadata.csv")
    pd.DataFrame({"tile_id": tile_ids, "class_label": class_labels}).to_csv(
        meta_out, index=True, index_label="idx"
    )
    print(f"Saved embedding metadata → {meta_out}")

    info = zarr_info(model_name)
    print(f"\nZarr info for {model_name}:")
    for k, v in info.items():
        print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet50", "dinov2", "clip", "all"],
        default="resnet50",
        help="Which model to embed with",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=TILES_DIR,
        help="Directory containing tile images (default: data/tiles/)",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV,
        help="metadata.csv produced by 01_download_data.py or 00_prepare_custom_data.py",
    )
    args = parser.parse_args()

    models = ["resnet50", "dinov2", "clip"] if args.model == "all" else [args.model]
    for m in models:
        build_for_model(m, tiles_dir=args.tiles_dir, metadata_csv=args.metadata_csv)

    print("\nDone! Next: python scripts/03_build_index.py --model <model>")


if __name__ == "__main__":
    main()
