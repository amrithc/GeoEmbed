"""Zarr-based embedding storage.

Why Zarr?
---------
Zarr stores N-dimensional arrays in *chunks* on disk. Each chunk is a
separate file (e.g. `0.0`, `1.0`, ...) inside the `.zarr/` directory.
This makes it:
  - Transparent: `ls embeddings.zarr/` shows individual chunk files
  - Cloud-compatible: swap DirectoryStore → S3Store with one line
  - Numpy-native: `zarr.open(path)[:]` returns a numpy array directly

For shape-(27000, 2048) float32 with chunk_size=512:
  - Each chunk: 512 × 2048 × 4 bytes ≈ 4MB
  - 53 chunk files total (52 full + 1 partial)

Design choice: We store ONLY the float32 embedding matrix in Zarr.
String metadata (tile_ids, class_labels) is stored in metadata.csv and
joined by integer index. This sidesteps zarr object-dtype quirks across
zarr v2/v3 versions.

Compare to alternatives:
  - HDF5: monolithic file, harder to stream; requires h5py
  - ChromaDB: fully-managed vector DB; hides chunking details (good for prod)
  - LanceDB: Arrow-native columnar; great for mixed embedding+metadata queries
"""

from pathlib import Path

import numpy as np
import zarr

from geoembded.config import EMBEDDINGS_DIR


def embedding_dir(model_name: str) -> Path:
    d = EMBEDDINGS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_zarr_store(
    model_name: str,
    n_samples: int,
    embed_dim: int,
    chunk_size: int = 512,
) -> zarr.Array:
    """Create an empty Zarr array for embeddings.

    Shape: (n_samples, embed_dim)
    Chunks: (chunk_size, embed_dim) — each chunk is one contiguous file on disk.
    """
    store_path = str(embedding_dir(model_name) / "embeddings.zarr")
    z = zarr.open_array(
        store_path,
        mode="w",
        shape=(n_samples, embed_dim),
        chunks=(chunk_size, embed_dim),
        dtype="float32",
    )
    return z


def write_embeddings(
    model_name: str,
    embeddings: np.ndarray,
) -> None:
    """Write the full embedding matrix to the existing Zarr array."""
    store_path = str(embedding_dir(model_name) / "embeddings.zarr")
    z = zarr.open_array(store_path, mode="r+")
    z[:] = embeddings.astype(np.float32)
    print(f"Wrote {embeddings.shape} embeddings → {store_path}")


def read_embeddings(model_name: str) -> np.ndarray:
    """Load the full float32 embedding matrix from Zarr.

    Returns:
        embeddings: float32 ndarray of shape (N, embed_dim)

    Tip: pair with metadata.csv (loaded separately) to get tile_ids/labels.
    The integer row index is the shared key.
    """
    store_path = str(embedding_dir(model_name) / "embeddings.zarr")
    if not Path(store_path).exists():
        raise FileNotFoundError(
            f"No embeddings found for model '{model_name}' at {store_path}. "
            f"Run: python scripts/02_build_embeddings.py --model {model_name}"
        )
    z = zarr.open_array(store_path, mode="r")
    return z[:]  # loads full array into numpy


def zarr_info(model_name: str) -> dict:
    """Return shape, dtype, chunk info for inspection in notebooks."""
    store_path = str(embedding_dir(model_name) / "embeddings.zarr")
    z = zarr.open_array(store_path, mode="r")
    return {
        "shape": z.shape,
        "dtype": str(z.dtype),
        "chunks": z.chunks,
        "nbytes_mb": z.nbytes / 1e6,
        "nchunks": z.nchunks,
        "store_path": store_path,
    }
