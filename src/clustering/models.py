from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PreparedMatrix:
    geo_codes: list[str]
    feature_names: list[str]
    X: np.ndarray
    X_scaled: np.ndarray
    pipeline: Pipeline


def prepare_feature_matrix(
    features_df: pd.DataFrame,
    *,
    geo_col: str = "geo_code",
    scaler: str = "robust",
    drop_missing_rate_gt: float = 0.98,
    drop_zero_variance: bool = True,
) -> PreparedMatrix:
    df = features_df.copy()
    geo_codes = df[geo_col].astype(str).tolist()
    X_df = df.drop(columns=[geo_col])

    # Ensure numeric
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # Drop features that are almost entirely missing (these become noise after imputation)
    missing_rate = X_df.isna().mean(axis=0)
    keep_cols = missing_rate[missing_rate <= float(drop_missing_rate_gt)].index.tolist()
    X_df = X_df[keep_cols]
    feature_names = X_df.columns.astype(str).tolist()

    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("scaler", RobustScaler(with_centering=True, with_scaling=True)))

    pipeline = Pipeline(steps)
    X_scaled = pipeline.fit_transform(X_df.to_numpy())

    # Drop zero variance logic needs to be handled carefully with feature names
    # Previous logic was post-scaling or post-imputation
    if drop_zero_variance:
        variances = np.nanvar(X_scaled, axis=0)
        keep = variances > 0.0
        X_scaled = X_scaled[:, keep]
        feature_names = np.array(feature_names)[keep].tolist()
        
        # NOTE: If we filter columns here, the pipeline object isn't strictly 
        # usable on new data without that same filter. 
        # For this refactor, we accept this limitation or would need a custom transformer.

    return PreparedMatrix(
        geo_codes=geo_codes,
        feature_names=feature_names,
        X=X_df.to_numpy(), # Original numeric data (with nans)
        X_scaled=X_scaled,
        pipeline=pipeline
    )


@dataclass(frozen=True)
class FitResult:
    labels: np.ndarray
    membership_prob: Optional[np.ndarray]
    model: Any


def fit_kmeans(
    X: np.ndarray,
    *,
    k: int,
    seed: int = 42,
    n_init: int = 20,
    init: str = "k-means++",
    max_iter: int = 300,
) -> FitResult:
    model = KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=int(n_init),
        init=init,
        max_iter=int(max_iter),
    )
    labels = model.fit_predict(X)
    return FitResult(labels=labels, membership_prob=None, model=model)


def fit_gmm(
    X: np.ndarray,
    *,
    n_components: int,
    seed: int = 42,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
) -> FitResult:
    model = GaussianMixture(
        n_components=int(n_components),
        random_state=int(seed),
        covariance_type=covariance_type,
        reg_covar=float(reg_covar),
    )
    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)
    membership = probs.max(axis=1) if probs is not None else None
    return FitResult(labels=labels, membership_prob=membership, model=model)
