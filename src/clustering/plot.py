from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def save_cluster_scatter_png(
    *,
    emb_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> bool:
    """
    Save a 2D scatter plot to PNG.

    - emb_df: columns [geo_code, emb_x, emb_y]
    - labels_df: columns [geo_code, cluster_id]
    Returns True if saved, False if matplotlib isn't available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt
    except Exception:
        return False

    df = emb_df.merge(labels_df[["geo_code", "cluster_id"]], on="geo_code", how="left")

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(df["emb_x"], df["emb_y"], c=df["cluster_id"], cmap="tab10", s=60, alpha=0.9)

    # Labels
    for _, row in df.iterrows():
        ax.annotate(
            str(row["geo_code"]),
            (float(row["emb_x"]), float(row["emb_y"])),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
            alpha=0.85,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="cluster_id")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True

