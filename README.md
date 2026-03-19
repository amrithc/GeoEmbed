# GeoEmbed

Geospatial tile similarity search using image embeddings and vector indexes.

Upload a satellite image (or type a text description) and retrieve the most visually similar tiles from your indexed database — all running locally on your machine, no API keys needed.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## What it does

| Feature | Description |
|---------|-------------|
| **Image search** | Upload any satellite tile → retrieve the most visually similar tiles |
| **Text search** | Type `"dense forest"` or `"river delta"` → find matching tiles via CLIP zero-shot |
| **Model comparison** | Swap between ResNet50, DINOv2, and CLIP at runtime |
| **Index comparison** | Flat (exact), IVF (cluster-based), HNSW (graph-based) FAISS indexes |
| **Bring your own data** | Index any folder of satellite or aerial images — not just EuroSAT |

---

## Two ways to get started

### Option A — Try it with EuroSAT (recommended for first run)

EuroSAT is a public dataset of 27,000 labeled 64×64 Sentinel-2 patches across 10
land-use classes. It takes 5 minutes to set up and immediately shows what the tool
can do: query a forest tile, get back forest tiles; query an industrial area, get
back industrial areas.

**[Jump to EuroSAT quickstart →](#eurosat-demo-quickstart)**

### Option B — Index your own images

Have your own satellite or aerial imagery? Point GeoEmbed at your folder and
build a searchable index in the same way.

**[Jump to custom data guide →](#bring-your-own-data)**

---

## EuroSAT demo quickstart

### 1. Create conda environment

```bash
mamba env create -f environment.yml
conda activate geoembded
```

> `faiss-cpu` is installed from conda-forge (not pip) to share OpenMP with PyTorch
> and avoid runtime conflicts on macOS.

### 2. Download EuroSAT (~90 MB)

```bash
python scripts/01_download_data.py
```

Downloads and flattens 27,000 labeled tiles to `data/tiles/` and writes `data/metadata.csv`.

### 3. Generate embeddings (~5–10 min per model on M2)

Start with DINOv2 for best results, or run all three to compare:

```bash
python scripts/02_build_embeddings.py --model dinov2    # best retrieval quality
python scripts/02_build_embeddings.py --model clip      # enables text search
python scripts/02_build_embeddings.py --model resnet50  # CNN baseline
```

First run downloads model weights (~330 MB DINOv2, ~600 MB CLIP) — cached after that.

### 4. Build FAISS indexes

```bash
python scripts/03_build_index.py --model all --index-type all
```

### 5. (Optional) Evaluate retrieval quality

```bash
python scripts/04_evaluate.py --model all --k 10
```

Expected precision@10: **DINOv2 ≈ 0.90 · CLIP ≈ 0.80 · ResNet50 ≈ 0.70**

### 6. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## Bring your own data

You can index any collection of satellite or aerial images. The pipeline is identical
to EuroSAT — you just swap in your own data at step 0.

### Folder structure

Organise images into one subfolder per class label:

```
my_images/
├── Forest/
│   ├── img001.jpg
│   └── img002.tif
├── Urban/
│   ├── img003.jpg
│   └── img004.png
└── Water/
    └── img005.jpg
```

The subfolder name becomes the `class_label`. If your images have no classes,
put them all in one subfolder (e.g. `my_images/unlabeled/`) — text search and
image search still work, evaluation just won't have ground-truth labels.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

### Prepare your data

```bash
python scripts/00_prepare_custom_data.py --source-dir /path/to/my_images
```

This copies all images to `data/tiles/` (converting to JPEG) and writes `data/metadata.csv`.
Prints a class distribution summary when done.

### Then run the same pipeline

```bash
python scripts/02_build_embeddings.py --model dinov2
python scripts/03_build_index.py --model dinov2 --index-type flat
streamlit run app/streamlit_app.py
```

### Custom output locations

If you want to keep multiple datasets indexed simultaneously:

```bash
# Prepare into a named directory
python scripts/00_prepare_custom_data.py \
    --source-dir /path/to/my_images \
    --tiles-dir data/my_tiles \
    --metadata-csv data/my_metadata.csv

# Embed from the named directory
python scripts/02_build_embeddings.py \
    --model dinov2 \
    --tiles-dir data/my_tiles \
    --metadata-csv data/my_metadata.csv
```

---

## Architecture

```
Your images
    │
    ▼
scripts/00_prepare_custom_data.py   (or 01_download_data.py for EuroSAT)
    │  data/tiles/   +   data/metadata.csv
    ▼
scripts/02_build_embeddings.py  ──►  embeddings/<model>/embeddings.zarr
                                      embeddings/<model>/metadata.csv
    ▼
scripts/03_build_index.py       ──►  embeddings/<model>/faiss_<type>.index
    │
    ▼
streamlit run app/streamlit_app.py
    │
    ├── Image query  →  embed with same model  →  FAISS search  →  top-K tiles
    └── Text query   →  CLIP text encoder      →  FAISS search  →  top-K tiles
```

### Embedding models

| Model | Dim | How it works | Best for |
|-------|-----|-------------|---------|
| **ResNet50** | 2048 | CNN avgpool features, ImageNet pretrained | Baseline; fast |
| **DINOv2** | 384 | ViT [CLS] token, self-supervised on 142M images | Best visual clustering |
| **CLIP** | 512 | Shared image+text space, trained on 400M pairs | Text search + image search |

### FAISS index types

| Index | Recall | Speed | Notes |
|-------|--------|-------|-------|
| **Flat** | 100% exact | ~10 ms | Brute force; best for <50K vectors |
| **IVF** | ~95% | ~1–2 ms | K-means clusters; needs training |
| **HNSW** | ~98% | ~0.5 ms | Graph-based; best speed/recall |

### Storage

Embeddings are stored as chunked float32 arrays in [Zarr](https://zarr.dev/) format.
Each `embeddings.zarr/` directory contains visible chunk files on disk — no black boxes.

---

## Project structure

```
GeoEmbed/
├── environment.yml                  # conda env (Python 3.11, conda-forge)
├── requirements.txt                 # pip fallback
├── geoembded/                       # core library
│   ├── config.py                    # paths, device, model configs
│   ├── data/
│   │   ├── downloader.py            # EuroSAT download + flatten
│   │   └── tile_dataset.py          # PyTorch Dataset (works with any tiles dir)
│   ├── embeddings/
│   │   ├── base.py                  # abstract EmbeddingModel
│   │   ├── resnet.py                # ResNet50 (2048-dim)
│   │   ├── dinov2.py                # DINOv2 ViT-S/14 (384-dim)
│   │   ├── clip_model.py            # CLIP ViT-B/32 (512-dim)
│   │   └── registry.py              # model name → class
│   ├── storage/
│   │   ├── zarr_store.py            # Zarr read/write
│   │   └── faiss_index.py           # FAISS build/save/load/search
│   └── search/
│       ├── image_search.py          # ImageSearchEngine
│       └── text_search.py           # TextSearchEngine (CLIP text → image index)
├── scripts/
│   ├── 00_prepare_custom_data.py    # prepare your own image dataset
│   ├── 01_download_data.py          # download EuroSAT demo dataset
│   ├── 02_build_embeddings.py       # embed all tiles → Zarr
│   ├── 03_build_index.py            # build FAISS index
│   ├── 04_evaluate.py               # precision@K evaluation
│   └── download_test_images.py      # fetch Sentinel-2 thumbnails for testing
└── app/
    └── streamlit_app.py             # Streamlit search UI
```

---

## Hardware

- **Apple Silicon M2/M3:** MPS auto-detected, all models run locally
- **CUDA GPU:** Auto-detected and used if available
- **CPU-only:** Works, ~3× slower than MPS for embedding generation

DINOv2 uses `batch_size=16` on MPS to stay within 8GB RAM.

---

## EuroSAT dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — 27,000 labeled 64×64 Sentinel-2 RGB patches, 10 land-use classes:

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

Downloaded automatically by `scripts/01_download_data.py` via `torchvision.datasets.EuroSAT`. Not included in this repo.

---

## Test query images

To download real Sentinel-2 thumbnails from outside EuroSAT (for testing the search app):

```bash
python scripts/download_test_images.py
```

Downloads 6 cloud-free (<5% cloud cover) scenes from globally diverse locations
(Amazon rainforest, Lake Victoria, Tokyo, Iowa farmland, Sahara, Greenland) via
the [Element84 Earth Search STAC API](https://earth-search.aws.element84.com/v1) — no authentication required.
