"""Central configuration — paths, model settings, device detection."""

import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TILES_DIR = DATA_DIR / "tiles"
METADATA_CSV = DATA_DIR / "metadata.csv"
EMBEDDINGS_DIR = ROOT / "embeddings"

# Ensure directories exist
for _d in [RAW_DIR, TILES_DIR, EMBEDDINGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Returns MPS on Apple Silicon, CUDA if available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ── Model configs ─────────────────────────────────────────────────────────────
# batch_size is conservative for MPS; increase on CUDA/CPU-only machines
MODEL_CONFIGS: dict[str, dict] = {
    "resnet50": {
        "embed_dim": 2048,
        "batch_size": 64 if DEVICE.type != "mps" else 64,
        "input_size": 224,
    },
    "dinov2": {
        "embed_dim": 384,        # dinov2_vits14 (ViT-Small, 21M params)
        "batch_size": 16,         # smaller batch to avoid MPS memory pressure
        "input_size": 224,
    },
    "clip": {
        "embed_dim": 512,         # ViT-B/32
        "batch_size": 64 if DEVICE.type != "mps" else 64,
        "input_size": 224,
    },
}

# ── FAISS index types ─────────────────────────────────────────────────────────
FAISS_INDEX_TYPES = ["flat", "ivf", "hnsw"]

# IVF clustering parameter — must have >= nlist*39 training vectors
IVF_NLIST = 100

# HNSW connectivity parameter — higher = better recall, more memory
HNSW_M = 32
