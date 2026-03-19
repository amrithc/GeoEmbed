"""FAISS index factory — build, save, load, and query similarity indexes.

FAISS Index Types
-----------------
Three index types are exposed, each with different speed/accuracy/memory trade-offs:

1. **Flat (IndexFlatIP)**
   - Brute-force exact cosine similarity (inner product on L2-normalised vectors)
   - O(N) per query — compares against every vector
   - 100% recall, no training required
   - Best for N < 50K or when accuracy must be perfect
   - For EuroSAT (27K vectors): queries finish in ~10ms

2. **IVF (IndexIVFFlat)**
   - Divides the vector space into `nlist` Voronoi clusters via k-means
   - At query time, only searches the `nprobe` nearest clusters (default 10)
   - Requires training: `index.train(embeddings)` before `index.add()`
   - Typical speedup: 10–50× over Flat at ~95% recall
   - Best for 50K–5M vectors
   - Pitfall: needs >= nlist*39 training vectors; add guard below

3. **HNSW (IndexHNSWFlat)**
   - Graph-based approximate nearest neighbour (Hierarchical Navigable Small World)
   - Navigates a multi-layer proximity graph; sub-linear query time
   - Best recall/speed trade-off; no training required
   - Higher memory than IVF (stores the graph structure)
   - Best for interactive, low-latency search over any dataset size

All embeddings must be L2-normalised before indexing when using Inner Product
similarity (so IP == cosine similarity). CLIP always normalises. ResNet/DINOv2
outputs are normalised inside build_index() below.
"""

from pathlib import Path

import faiss
import numpy as np

from geoembded.config import EMBEDDINGS_DIR, IVF_NLIST, HNSW_M


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row so cosine similarity == inner product."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    return (x / norms).astype(np.float32)


def build_index(
    embeddings: np.ndarray,
    index_type: str = "flat",
    nlist: int = IVF_NLIST,
    nprobe: int = 10,
    M: int = HNSW_M,
) -> faiss.Index:
    """Build a FAISS index from an embedding matrix.

    Args:
        embeddings: float32 array of shape (N, embed_dim).
        index_type: One of 'flat', 'ivf', 'hnsw'.
        nlist: Number of IVF clusters (only used when index_type='ivf').
        nprobe: Number of clusters to search at query time (IVF only).
        M: HNSW connectivity parameter (HNSW only).

    Returns:
        Trained and populated FAISS index.
    """
    embeddings = _l2_normalize(embeddings)
    n, d = embeddings.shape

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

    elif index_type == "ivf":
        min_train = nlist * 39
        if n < min_train:
            raise ValueError(
                f"IVF requires >= {min_train} training vectors (nlist={nlist}×39). "
                f"Got {n}. Reduce nlist or use index_type='flat'."
            )
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

    else:
        raise ValueError(f"Unknown index_type '{index_type}'. Choose: flat, ivf, hnsw")

    return index


def save_index(index: faiss.Index, model_name: str, index_type: str) -> Path:
    """Save a FAISS index to disk."""
    path = EMBEDDINGS_DIR / model_name / f"faiss_{index_type}.index"
    faiss.write_index(index, str(path))
    print(f"Saved FAISS {index_type} index → {path}")
    return path


def load_index(model_name: str, index_type: str = "flat") -> faiss.Index:
    """Load a FAISS index from disk."""
    path = EMBEDDINGS_DIR / model_name / f"faiss_{index_type}.index"
    if not path.exists():
        raise FileNotFoundError(
            f"No FAISS index at {path}. "
            f"Run: python scripts/03_build_index.py --model {model_name} --index-type {index_type}"
        )
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    query: np.ndarray,
    k: int = 10,
    query_already_normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index for the k nearest neighbours.

    Args:
        index: A loaded FAISS index.
        query: 1D or 2D float32 array (single vector or batch).
        k: Number of results to return.
        query_already_normalized: Skip L2-norm if embedder already normalises.

    Returns:
        scores: cosine similarity scores, shape (n_queries, k)
        indices: integer row indices into the original embedding matrix, shape (n_queries, k)
    """
    if query.ndim == 1:
        query = query[np.newaxis, :]

    if not query_already_normalized:
        query = _l2_normalize(query)

    scores, indices = index.search(query, k)
    return scores, indices
