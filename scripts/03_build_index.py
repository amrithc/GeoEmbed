#!/usr/bin/env python
"""Stage 3: Build FAISS similarity index from stored Zarr embeddings.

Usage:
    python scripts/03_build_index.py --model resnet50 --index-type flat
    python scripts/03_build_index.py --model clip --index-type hnsw
    python scripts/03_build_index.py --model all --index-type all
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoembded.config import FAISS_INDEX_TYPES
from geoembded.storage.zarr_store import read_embeddings
from geoembded.storage.faiss_index import build_index, save_index


def build_for_model_and_type(model_name: str, index_type: str) -> None:
    print(f"\n  Building {index_type.upper()} index for {model_name}...")
    embeddings = read_embeddings(model_name)
    print(f"  Loaded embeddings: {embeddings.shape}")

    t0 = time.perf_counter()
    index = build_index(embeddings, index_type=index_type)
    elapsed = time.perf_counter() - t0

    save_index(index, model_name, index_type)
    print(f"  Built in {elapsed:.2f}s — total vectors: {index.ntotal}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "dinov2", "clip", "all"], default="resnet50")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw", "all"], default="flat")
    args = parser.parse_args()

    models = ["resnet50", "dinov2", "clip"] if args.model == "all" else [args.model]
    index_types = FAISS_INDEX_TYPES if args.index_type == "all" else [args.index_type]

    for m in models:
        print(f"\n{'='*60}\nModel: {m}\n{'='*60}")
        for idx_type in index_types:
            try:
                build_for_model_and_type(m, idx_type)
            except FileNotFoundError as e:
                print(f"  SKIP {idx_type}: {e}")
            except ValueError as e:
                print(f"  SKIP {idx_type}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
