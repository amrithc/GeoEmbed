"""Phase 2: Text-to-image search via CLIP's shared embedding space.

How it works
------------
CLIP trains image and text encoders with the same contrastive objective,
so both output vectors in the same 512-dim space. A text embedding like
"a satellite image of a forest" will be close to forest tile embeddings
because CLIP saw millions of such (image, caption) pairs during training.

Key insight: the FAISS index built from *image* embeddings (Phase 1) can be
queried directly with a *text* embedding — no new index or training needed.

Usage:
    engine = TextSearchEngine(index_type="flat")
    results = engine.search("show me tiles with lots of buildings", k=10)

Prompt tips for EuroSAT
-----------------------
- "farmland and agricultural fields"      → AnnualCrop, PermanentCrop
- "dense urban residential area"          → Residential
- "industrial buildings and warehouses"   → Industrial
- "forest and dense trees"                → Forest
- "river or water body"                   → River, SeaLake
- "highway or road"                       → Highway
- "grass and meadow"                      → HerbaceousVegetation, Pasture
"""

from pathlib import Path

from geoembded.search.image_search import ImageSearchEngine, SearchResult


class TextSearchEngine(ImageSearchEngine):
    """Searches the CLIP image index using a text query.

    Requires model_name='clip' (text embeddings only available from CLIP).
    """

    def __init__(self, index_type: str = "flat") -> None:
        super().__init__(model_name="clip", index_type=index_type)

    def search(self, query_text: str, k: int = 10) -> list[SearchResult]:
        """Find the k tiles most semantically matching the text description.

        Args:
            query_text: Natural language description (e.g. "dense forest").
            k: Number of results to return.

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        self._load()

        query_vec = self._embedder.embed_text(query_text)

        from geoembded.storage.faiss_index import search as faiss_search
        scores, indices = faiss_search(self._index, query_vec, k=k)
        scores = scores[0]
        indices = indices[0]

        from geoembded.config import TILES_DIR
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
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
