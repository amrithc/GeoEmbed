#!/usr/bin/env python
"""Download Sentinel-2 scene thumbnails from Earth Search STAC API as test images.

These are real satellite images from locations NOT in EuroSAT (which covers
European land use). These thumbnails come from globally diverse scenes —
Amazon rainforest, Lake Victoria, Tokyo, Iowa farmland, Sahara, Greenland.

Usage:
    python scripts/download_test_images.py

Output: test_images/<label>_<scene_id>.jpg
"""

import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

STAC_URL = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"
OUT = Path(__file__).parent.parent / "test_images"
OUT.mkdir(exist_ok=True)

# Each entry: (label, [lon_min, lat_min, lon_max, lat_max])
SCENES = [
    ("forest",   [-61.0, -4.0,  -59.0, -2.0]),   # Amazon rainforest, Brazil
    ("water",    [ 32.0, -1.0,   34.0,  1.0]),   # Lake Victoria, East Africa
    ("urban",    [139.5, 35.5,  140.0, 36.0]),   # Tokyo, Japan
    ("farmland", [-94.0, 41.0,  -92.0, 43.0]),   # Iowa croplands, USA
    ("desert",   [  9.0, 24.0,   11.0, 26.0]),   # Sahara, Libya
    ("snow",     [-42.0, 71.0,  -38.0, 73.0]),   # Greenland ice sheet
]


def fetch_thumbnail(label: str, bbox: list[float]) -> Path | None:
    """Query Earth Search for the clearest scene in bbox (cloud cover < 5%) and download its thumbnail."""
    params = {
        "bbox": ",".join(str(x) for x in bbox),
        "limit": 20,           # fetch candidates so we can pick the clearest
        "sortby": "-properties.datetime",
        "filter": "eo:cloud_cover<5",
        "filter-lang": "cql2-text",
    }

    try:
        resp = requests.get(STAC_URL, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("features", [])
    except Exception as e:
        print(f"  {label}: STAC query failed — {e}")
        return None

    if not items:
        print(f"  {label}: no scenes with cloud cover <5% found")
        return None

    # Pick the scene with lowest cloud cover among candidates
    item = min(items, key=lambda i: i.get("properties", {}).get("eo:cloud_cover", 999))
    cloud_pct = item.get("properties", {}).get("eo:cloud_cover", "?")
    print(f"  {label}: best scene has {cloud_pct}% cloud cover")
    scene_id = item["id"][:16]  # truncate for readable filename

    # Try thumbnail asset first, fall back to overview
    assets = item.get("assets", {})
    thumb_url = None
    for key in ("thumbnail", "overview", "rendered_preview", "visual"):
        if key in assets:
            thumb_url = assets[key].get("href")
            break

    if not thumb_url:
        print(f"  {label}: no thumbnail asset in scene {scene_id}")
        return None

    try:
        r = requests.get(thumb_url, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        # Resize to 256×256 — large enough to show detail, small enough to load quickly
        img = img.resize((256, 256), Image.LANCZOS)
        out_path = OUT / f"{label}_{scene_id}.jpg"
        img.save(out_path, "JPEG", quality=90)
        print(f"  ✓ {label:10s}  {out_path.name}  ({img.size[0]}×{img.size[1]})")
        return out_path
    except Exception as e:
        print(f"  {label}: download failed — {e}")
        return None


def main() -> None:
    print(f"Downloading test images to {OUT}/\n")
    downloaded = []
    for label, bbox in SCENES:
        path = fetch_thumbnail(label, bbox)
        if path:
            downloaded.append(path)

    print(f"\n{len(downloaded)}/{len(SCENES)} test images saved to {OUT}/")
    if downloaded:
        print("\nFiles:")
        for p in downloaded:
            print(f"  {p.name}")
        print(f"\nUpload any of these to the Streamlit app to test search.")


if __name__ == "__main__":
    main()
