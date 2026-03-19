"""Visualisation helpers — display search result grids in notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def show_results(results, query_image=None, cols: int = 5, figsize_per_tile: float = 2.0):
    """Display a grid of search results with similarity scores.

    Args:
        results: List of SearchResult from ImageSearchEngine or TextSearchEngine.
        query_image: Optional PIL Image or path to show as the query.
        cols: Number of columns in the grid.
        figsize_per_tile: Size of each tile in inches.
    """
    n = len(results)
    rows = (n + cols - 1) // cols

    offset = 1 if query_image is not None else 0
    total_slots = n + offset

    fig_cols = min(cols + offset, total_slots)
    fig_rows = (total_slots + fig_cols - 1) // fig_cols

    fig, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(figsize_per_tile * fig_cols, figsize_per_tile * fig_rows),
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    ax_idx = 0

    if query_image is not None:
        if isinstance(query_image, (str, Path)):
            query_image = Image.open(query_image).convert("RGB")
        axes[ax_idx].imshow(query_image)
        axes[ax_idx].set_title("Query", fontsize=8, color="blue", fontweight="bold")
        axes[ax_idx].axis("off")
        ax_idx += 1

    for r in results:
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        if Path(r.image_path).exists():
            img = Image.open(r.image_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"#{r.rank} {r.class_label}\n{r.score:.3f}", fontsize=7)
        ax.axis("off")
        ax_idx += 1

    # Hide unused axes
    for ax in axes[ax_idx:]:
        ax.axis("off")

    plt.tight_layout()
    return fig
