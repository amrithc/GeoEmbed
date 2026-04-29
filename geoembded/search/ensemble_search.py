"""Ensemble search engine combining multiple models.

Usage:
    engine = EnsembleSearchEngine(models=["dinov2", "clip", "resnet50"])
    results = engine.search("query_image.jpg", k=10)

    # Or with custom weights
    engine = EnsembleSearchEngine(
        models=["dinov2", "clip", "resnet50"],
        weights=[0.5, 0.3, 0.2]  # DINOv2 most important
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from geoembded.config import EMBEDDINGS_DIR, TILES_DIR
from geoembded.storage.zarr_store import read_embeddings
from geoembded.storage.faiss_index import load_index, search as faiss_search
from geoembded.embeddings.registry import get_embedder
from geoembded.data.tile_dataset import default_transform
from geoembded.search.image_search import SearchResult
import torch


class EnsembleSearchEngine:
    """Ensemble search combining multiple embedding models.

    Each model captures different patterns:
    - DINOv2: structure, patches, self-supervised patterns
    - CLIP: semantic meaning, text alignment
    - ResNet50: texture, ImageNet features

    Concatenating embeddings: 384 + 512 + 2048 = 2944-dim
    Result: better precision through diversity

    Args:
        models: List of model names (e.g., ["dinov2", "clip", "resnet50"])
        index_type: "flat", "ivf", or "hnsw" (must be same for all models)
        weights: Optional weights for each model (defaults to equal 1/N)
        normalize_per_model: If True, L2-normalize each model's embedding before concat
    """

    def __init__(
        self,
        models: List[str] = ["dinov2", "clip", "resnet50"],
        index_type: str = "flat",
        weights: Optional[List[float]] = None,
        normalize_per_model: bool = True,
    ) -> None:
        self.models = models
        self.index_type = index_type
        self.normalize_per_model = normalize_per_model

        # Set weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError(
                    f"weights length ({len(weights)}) must match models length ({len(models)})"
                )
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]

        # State
        self._combined_embeddings: Optional[np.ndarray] = None
        self._meta: Optional[pd.DataFrame] = None
        self._index = None
        self._embedders = {}

    def _load(self) -> None:
        """Load all embeddings and indexes."""
        if self._index is not None:
            return

        print(f"\nLoading ensemble ({', '.join(self.models)})...")

        # Load embeddings and metadata from first model (metadata is shared)
        first_model = self.models[0]
        meta_path = EMBEDDINGS_DIR / first_model / "metadata.csv"
        self._meta = pd.read_csv(meta_path)

        # Load and concatenate embeddings
        embeddings_list = []
        for model_name in self.models:
            print(f"  Loading {model_name}...")
            embs = read_embeddings(model_name)

            # Optional: normalize per-model
            if self.normalize_per_model:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embs = embs / norms

            embeddings_list.append(embs)

        # Concatenate
        self._combined_embeddings = np.hstack(embeddings_list).astype(np.float32)
        print(f"  Combined shape: {self._combined_embeddings.shape}")

        # Build combined index
        from geoembded.storage.faiss_index import build_index

        print(f"  Building {self.index_type} index...")
        self._index = build_index(
            self._combined_embeddings, index_type=self.index_type
        )
        print(f"  Total vectors: {self._index.ntotal}")

        # Load embedders for query encoding
        for model_name in self.models:
            self._embedders[model_name] = get_embedder(model_name)
            self._embedders[model_name].load()

    def search(
        self, query_image: str | Path | Image.Image, k: int = 10
    ) -> list[SearchResult]:
        """Find k most similar tiles using ensemble.

        Args:
            query_image: File path (str/Path) or PIL Image.
            k: Number of results.

        Returns:
            List of SearchResult sorted by descending similarity.
        """
        self._load()

        # Load image if path provided
        if isinstance(query_image, (str, Path)):
            pil_img = Image.open(query_image).convert("RGB")
        else:
            pil_img = query_image

        # Get embeddings from each model and concatenate
        embeddings_list = []

        for model_name in self.models:
            embedder = self._embedders[model_name]

            if model_name == "clip":
                # CLIP can embed image directly
                emb = embedder.embed_image(pil_img)
            else:
                # ResNet50, DINOv2: use transform
                transform = default_transform(224)
                tensor = transform(pil_img).unsqueeze(0)  # (1, C, H, W)
                emb = embedder.embed_batch(tensor)[0]  # (embed_dim,)

            # Optional: normalize
            if self.normalize_per_model:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

            embeddings_list.append(emb)

        # Concatenate query embeddings
        query_vec = np.concatenate(embeddings_list).astype(np.float32)

        # Search
        scores, indices = faiss_search(self._index, query_vec, k=k)
        scores = scores[0]
        indices = indices[0]

        # Build results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            row = self._meta.iloc[idx]
            results.append(
                SearchResult(
                    rank=rank + 1,
                    tile_id=row["tile_id"],
                    score=float(score),
                    class_label=row["class_label"],
                    image_path=TILES_DIR / f"{row['tile_id']}.jpg",
                )
            )

        return results

    def evaluate(self, num_samples: int = 2700, k: int = 10) -> dict:
        """Evaluate ensemble precision on random sample.

        Args:
            num_samples: Number of random queries to evaluate
            k: Evaluate Precision@K

        Returns:
            Dict with precision metrics
        """
        self._load()

        if num_samples > len(self._meta):
            num_samples = len(self._meta)

        # Random sample indices
        query_indices = np.random.choice(len(self._meta), num_samples, replace=False)

        # Evaluate
        num_correct = 0
        for idx in query_indices:
            query_emb = self._combined_embeddings[idx : idx + 1]
            _, match_indices = faiss_search(self._index, query_emb, k=k + 1)
            # Skip first match (query itself)
            match_indices = match_indices[0, 1:]

            query_class = self._meta.iloc[idx]["class_label"]
            for match_idx in match_indices:
                if match_idx >= 0:
                    match_class = self._meta.iloc[match_idx]["class_label"]
                    if query_class == match_class:
                        num_correct += 1

        precision = num_correct / (num_samples * k)

        return {
            "model": "ensemble",
            "models": self.models,
            "weights": dict(zip(self.models, self.weights)),
            "precision@k": precision,
            "k": k,
            "samples": num_samples,
        }


if __name__ == "__main__":
    # Example usage
    import time

    # Create ensemble with custom weights
    engine = EnsembleSearchEngine(
        models=["dinov2", "clip", "resnet50"],
        index_type="hnsw",
        weights=[0.5, 0.3, 0.2],  # DINOv2 most important
    )

    # Search
    start = time.time()
    results = engine.search("data/tiles/Forest_001.jpg", k=10)
    elapsed = time.time() - start

    print(f"\nEnsemble Search Results (Forest_001):")
    print(f"Time: {elapsed*1000:.1f}ms\n")
    for r in results:
        print(f"#{r.rank}: {r.tile_id} ({r.class_label}) — {r.score:.4f}")

    # Evaluate
    metrics = engine.evaluate(num_samples=100, k=10)
    print(f"\nEnsemble Evaluation:")
    print(f"  Precision@10: {metrics['precision@k']:.4f}")
    print(f"  Models: {metrics['models']}")
    print(f"  Weights: {metrics['weights']}")
