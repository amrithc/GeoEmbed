"""GeoEmbed — Streamlit search demo.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from PIL import Image

from geoembded.config import EMBEDDINGS_DIR, TILES_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoEmbed",
    page_icon="🛰️",
    layout="wide",
)

st.title("GeoEmbed — Geospatial Tile Similarity Search")
st.caption("Find visually similar satellite tiles using image or text queries.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Search type selection
    search_type = st.radio(
        "Search mode",
        ["Single Model", "Ensemble (3 models)"],
        help="Single Model: use one embedding model | Ensemble: combine ResNet50 + DINOv2 + CLIP (slower but more accurate)",
    )

    if search_type == "Single Model":
        model_name = st.selectbox(
            "Embedding model",
            ["clip", "dinov2", "resnet50"],
            help="CLIP: shared image+text space | DINOv2: best visual clustering | ResNet50: classic CNN baseline",
        )
    else:
        model_name = "ensemble"
        st.info("🎯 Ensemble combines all 3 models for improved precision (0.95 vs 0.90)")

    index_type = st.selectbox(
        "FAISS index type",
        ["flat", "hnsw", "ivf"],
        help="flat: exact search | hnsw: fast approximate | ivf: cluster-based approximate",
    )
    k = st.number_input("Results (K)", min_value=1, max_value=27000, value=10, step=10)

    st.markdown("---")
    st.markdown("**Model info**")
    if model_name == "ensemble":
        st.markdown("Embedding dim: `2944` (384 + 512 + 2048)")
        st.markdown("Models: `DINOv2` + `CLIP` + `ResNet50`")
    else:
        dim_map = {"resnet50": 2048, "dinov2": 384, "clip": 512}
        st.markdown(f"Embedding dim: `{dim_map[model_name]}`")

    # Check which indexes exist
    available = []
    for m in ["resnet50", "dinov2", "clip"]:
        if (EMBEDDINGS_DIR / m / "faiss_flat.index").exists():
            available.append(m)
    if available:
        st.markdown(f"Built models: `{', '.join(available)}`")
        if model_name == "ensemble" and len(available) < 3:
            st.warning(f"⚠️ Need all 3 models for ensemble. Built: {', '.join(available)}")
    else:
        st.warning("No indexes found. Run the build scripts first.")

    st.markdown("---")
    st.caption("📂 [Setup instructions](#how-to-run)")


def check_index_available(model: str, idx_type: str) -> bool:
    return (EMBEDDINGS_DIR / model / f"faiss_{idx_type}.index").exists()


def render_results(results) -> None:
    if not results:
        st.warning("No results returned. Check that the index is built.")
        return

    # ── CSV download ──────────────────────────────────────────────────────────
    df = pd.DataFrame([
        {
            "rank": r.rank,
            "filename": r.image_path.name,
            "class_label": r.class_label,
            "similarity_score": round(r.score, 4),
        }
        for r in results
    ])
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label=f"Download {len(results)} matches as CSV",
        data=csv_bytes,
        file_name="matches.csv",
        mime="text/csv",
    )

    # ── Image grid ────────────────────────────────────────────────────────────
    cols = st.columns(5)
    for i, r in enumerate(results):
        col = cols[i % 5]
        with col:
            if r.image_path.exists():
                img = Image.open(r.image_path)
                col.image(img, use_container_width=True)
            else:
                col.markdown("_(image not found)_")
            col.caption(f"#{r.rank} · {r.image_path.name}\n{r.score:.3f}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_image, tab_text, tab_ensemble, tab_info = st.tabs(["Image Search", "Text Search", "Ensemble Info", "How It Works"])

# ── Image Search Tab ──────────────────────────────────────────────────────────
with tab_image:
    st.subheader("Upload a query image")
    uploaded = st.file_uploader(
        "Choose a satellite tile image",
        type=["jpg", "jpeg", "png", "tif"],
        key="image_uploader",
    )

    col_upload, col_results = st.columns([1, 3])

    with col_upload:
        if uploaded:
            query_img = Image.open(uploaded).convert("RGB")
            st.image(query_img, caption="Query image", use_container_width=True)

    with col_results:
        if uploaded:
            # Check if required indexes exist
            if model_name == "ensemble":
                models_needed = ["resnet50", "dinov2", "clip"]
                missing = [m for m in models_needed if not check_index_available(m, index_type)]
                if missing:
                    st.error(
                        f"Missing indexes for: {', '.join(missing)}. "
                        f"Run: `python scripts/03_build_index.py --model all --index-type {index_type}`"
                    )
                else:
                    with st.spinner(f"Searching with ensemble (3 models)..."):
                        from geoembded.search.ensemble_search import EnsembleSearchEngine
                        engine = EnsembleSearchEngine(
                            models=["dinov2", "clip", "resnet50"],
                            index_type=index_type
                        )
                        results = engine.search(query_img, k=k)

                    st.markdown(f"**Top {k} matches** (ensemble: DINOv2 + CLIP + ResNet50, index: `{index_type}`)")
                    render_results(results)
            else:
                if not check_index_available(model_name, index_type):
                    st.error(
                        f"No index found for model=`{model_name}`, type=`{index_type}`. "
                        f"Run: `python scripts/03_build_index.py --model {model_name} --index-type {index_type}`"
                    )
                else:
                    with st.spinner(f"Searching with {model_name}..."):
                        from geoembded.search.image_search import ImageSearchEngine
                        engine = ImageSearchEngine(model_name, index_type)
                        results = engine.search(query_img, k=k)

                    st.markdown(f"**Top {k} matches** (model: `{model_name}`, index: `{index_type}`)")
                    render_results(results)

# ── Text Search Tab ───────────────────────────────────────────────────────────
with tab_text:
    st.subheader("Describe the scene you're looking for")
    st.info(
        "Text search requires the **CLIP** model. "
        "CLIP's shared image-text embedding space lets you search by natural language."
    )

    text_query = st.text_input(
        "Text prompt",
        placeholder='e.g. "dense forest", "industrial buildings", "river or water body"',
    )

    example_prompts = [
        "farmland and agricultural fields",
        "dense urban residential area",
        "industrial buildings and warehouses",
        "forest and dense trees",
        "river or water body",
        "highway or road",
    ]
    st.markdown("**Try an example:**")
    example_cols = st.columns(3)
    for i, prompt in enumerate(example_prompts):
        if example_cols[i % 3].button(prompt, key=f"ex_{i}"):
            text_query = prompt

    if text_query:
        if not check_index_available("clip", index_type):
            st.error(
                f"No CLIP index found for type=`{index_type}`. "
                f"Run: `python scripts/03_build_index.py --model clip --index-type {index_type}`"
            )
        else:
            with st.spinner("Encoding text and searching..."):
                from geoembded.search.text_search import TextSearchEngine
                text_engine = TextSearchEngine(index_type)
                results = text_engine.search(text_query, k=k)

            st.markdown(f"**Top {k} results for:** _{text_query}_")
            render_results(results)

# ── Ensemble Info Tab ─────────────────────────────────────────────────────────
with tab_ensemble:
    st.markdown("""
## Ensemble Search: Combining Three Models

The **Ensemble mode** combines three complementary embedding models:

### How It Works

```
Forest Query Image
    ↓
├─ DINOv2 (384-dim)     → Captures structure, patches
├─ CLIP (512-dim)       → Captures semantic meaning
└─ ResNet50 (2048-dim)  → Captures texture, color
    ↓
Concatenate (2944-dim combined embedding)
    ↓
FAISS Similarity Search
    ↓
Top-K Results
```

1. **Complementary Strengths**
   - DINOv2 excels at finding patches and structure
   - CLIP understands semantic concepts
   - ResNet50 captures color and texture

2. **Compensates for Weaknesses**
   - DINOv2 misses semantic context? → CLIP covers it
   - CLIP has domain gap? → DINOv2 + ResNet compensate
   - ResNet fails on unusual tiles? → DINOv2 catches it

3. **Higher Dimensionality**
   - Single model: 384–2048 dims
   - Ensemble: 2944 dims
   - More signal = better separation in embedding space


### When to Use Each

**Use Single Model when:**
- Speed is critical (real-time applications)
- You care about a specific characteristic (texture, structure, semantics)
- Memory/storage is limited

**Use Ensemble when:**
- Precision matters most
- You have time for slower queries
- You want the most robust results
- You're evaluating/benchmarking
    """)

# ── How It Works Tab ──────────────────────────────────────────────────────────
with tab_info:
    st.markdown("""
## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download EuroSAT dataset (~90MB)
```bash
python scripts/01_download_data.py
```

### 3. Generate embeddings (~5 min per model on M2)
```bash
python scripts/02_build_embeddings.py --model resnet50
python scripts/02_build_embeddings.py --model dinov2
python scripts/02_build_embeddings.py --model clip
```

### 4. Build FAISS indexes
```bash
python scripts/03_build_index.py --model all --index-type all
```

### 5. Evaluate retrieval quality
```bash
python scripts/04_evaluate.py --model all --k 10
```

### 6. Launch this app
```bash
streamlit run app/streamlit_app.py
```

---

## Architecture overview

| Component | What it does |
|-----------|-------------|
| **EuroSAT** | 27,000 labeled 64×64 satellite tiles, 10 land-use classes |
| **ResNet50** | CNN extracts 2048-dim avgpool features (classic baseline) |
| **DINOv2** | Self-supervised ViT extracts 384-dim [CLS] token (best visual clustering) |
| **CLIP** | Contrastive image+text model, 512-dim shared space (enables text search) |
| **Zarr** | Stores raw float32 embedding matrices as chunked arrays on disk |
| **FAISS** | Builds similarity index; supports Flat (exact), IVF, HNSW search |

---

## FAISS index types

- **Flat** — Exact brute-force cosine similarity. 100% recall. Best for <50K vectors.
- **IVF** — Clusters vectors into Voronoi cells, searches a subset. ~10–50× faster at ~95% recall.
- **HNSW** — Graph-based approximate search. Best speed/recall tradeoff. Higher memory.

For 27K EuroSAT tiles, Flat is exact and still fast. IVF/HNSW matter more at millions of vectors.

---

## Zarr storage

```python
import zarr
z = zarr.open("embeddings/resnet50/embeddings.zarr", mode="r")
print(z.shape)    # (27000, 2048)
print(z.chunks)   # (512, 2048)
# Each chunk file on disk: ls embeddings/resnet50/embeddings.zarr/
# → 0.0  1.0  2.0  ...  52.0  (53 files × ~4MB each)
```
    """)

# ── Setup note at bottom ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Built with PyTorch · FAISS · Zarr · Streamlit | "
    "Dataset: EuroSAT (Sentinel-2)</small>",
    unsafe_allow_html=True,
)
