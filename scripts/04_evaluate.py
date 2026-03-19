#!/usr/bin/env python
"""Stage 4: Evaluate retrieval precision@K across models.

Uses class labels as ground truth:
  For each query tile, search top-K results.
  Precision@K = (results with same class label) / K

Expected ranking on EuroSAT: DINOv2 > CLIP > ResNet50

Usage:
    python scripts/04_evaluate.py --model all --k 10 --n-queries 500
    python scripts/04_evaluate.py --model resnet50 --k 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoembded.config import EMBEDDINGS_DIR
from geoembded.storage.zarr_store import read_embeddings
from geoembded.storage.faiss_index import load_index, search


def precision_at_k(
    embeddings: np.ndarray,
    class_labels: list[str],
    index,
    k: int,
    n_queries: int,
) -> float:
    """Compute mean precision@K over a random sample of queries."""
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(embeddings), size=min(n_queries, len(embeddings)), replace=False)

    precisions = []
    for qi in query_indices:
        query_vec = embeddings[qi]
        scores, indices = search(index, query_vec, k=k + 1)  # +1 to exclude self
        result_indices = indices[0]

        # Remove self-match (the query tile itself)
        result_indices = [i for i in result_indices if i != qi][:k]

        query_label = class_labels[qi]
        correct = sum(1 for i in result_indices if class_labels[i] == query_label)
        precisions.append(correct / k)

    return float(np.mean(precisions))


def evaluate_model(model_name: str, index_type: str, k: int, n_queries: int) -> dict:
    print(f"\nEvaluating {model_name} ({index_type} index)...")

    embeddings = read_embeddings(model_name)

    meta_path = EMBEDDINGS_DIR / model_name / "metadata.csv"
    if not meta_path.exists():
        print(f"  No metadata.csv found at {meta_path}, skipping")
        return {}

    meta = pd.read_csv(meta_path)
    class_labels = meta["class_label"].tolist()

    index = load_index(model_name, index_type)
    p_at_k = precision_at_k(embeddings, class_labels, index, k=k, n_queries=n_queries)

    result = {
        "model": model_name,
        "index_type": index_type,
        f"precision@{k}": round(p_at_k, 4),
        "n_queries": min(n_queries, len(embeddings)),
    }
    print(f"  precision@{k} = {p_at_k:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "dinov2", "clip", "all"], default="all")
    parser.add_argument("--index-type", default="flat")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=500)
    args = parser.parse_args()

    models = ["resnet50", "dinov2", "clip"] if args.model == "all" else [args.model]
    results = []

    for m in models:
        try:
            r = evaluate_model(m, args.index_type, args.k, args.n_queries)
            if r:
                results.append(r)
        except FileNotFoundError as e:
            print(f"  SKIP {m}: {e}")

    if results:
        print(f"\n{'='*50}")
        print("Results Summary")
        print(f"{'='*50}")
        df = pd.DataFrame(results).sort_values(f"precision@{args.k}", ascending=False)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
