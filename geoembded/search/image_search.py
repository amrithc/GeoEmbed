"""Image-to-image search engine.

Usage:
    engine = ImageSearchEngine("clip", "flat")
    results = engine.search("data/tiles/Forest_abc123.jpg", k=10)
    for r in results:
        print(r.tile_id, r.score, r.class_label, r.image_path)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geoembded.config import EMBEDDINGS_DIR, TILES_DIR
from geoembded.embeddings.registry import get_embedder
from geoembded.storage.zarr_store import read_embeddings
from geoembded.storage.faiss_index import load_index, search


@dataclass
class SearchResult:
    rank: int
    tile_id: str
    score: float
    class_label: str
    image_path: Path


class ImageSearchEngine:
    """Searches the FAISS index for tiles visually similar to a query image.

    Args:
        model_name: 'resnet50', 'dinov2', or 'clip'
        index_type: 'flat', 'ivf', or 'hnsw'
    """

    def __init__(self, model_name: str = "clip", index_type: str = "flat") -> None:
        self.model_name = model_name
        self.index_type = index_type
        self._embeddings: np.ndarray | None = None
        self._meta: pd.DataFrame | None = None
        self._index = None
        self._embedder = None

    def _load(self) -> None:
        if self._index is not None:
            return

        self._embeddings = read_embeddings(self.model_name)

        meta_path = EMBEDDINGS_DIR / self.model_name / "metadata.csv"
        self._meta = pd.read_csv(meta_path)

        self._index = load_index(self.model_name, self.index_type)

        self._embedder = get_embedder(self.model_name)
        self._embedder.load()

    def search(self, query_image: str | Path | Image.Image, k: int = 10) -> list[SearchResult]:
        """Find the k most similar tiles to a query image.

        Args:
            query_image: File path (str/Path) or PIL Image.
            k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        self._load()

        if isinstance(query_image, (str, Path)):
            pil_img = Image.open(query_image).convert("RGB")
        else:
            pil_img = query_image

        # Use model-specific single-image embedding
        if self.model_name == "clip":
            query_vec = self._embedder.embed_image(pil_img)
        else:
            # For ResNet50 and DINOv2: apply the default transform and embed
            from geoembded.data.tile_dataset import default_transform
            import torch
            transform = default_transform(224)
            tensor = transform(pil_img).unsqueeze(0)  # (1, C, H, W)
            query_vec = self._embedder.embed_batch(tensor)[0]  # (embed_dim,)

        scores, indices = search(self._index, query_vec, k=k)
        scores = scores[0]
        indices = indices[0]

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
