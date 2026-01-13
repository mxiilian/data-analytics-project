from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def save_run_results(
    out_dir: Path,
    labels_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> None:
    """
    Saves clustering results to CSV files in the specified directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save labels
    labels_path = out_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    logger.info("Saved cluster labels to %s", labels_path)

    # Save profiles
    profiles_path = out_dir / "profiles.csv"
    profiles_df.to_csv(profiles_path, index=False)
    logger.info("Saved feature profiles to %s", profiles_path)
