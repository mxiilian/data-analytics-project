from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass(frozen=True)
class PreparedMatrix:
    geo_codes: list[str]
    feature_names: list[str]
    X: np.ndarray
    X_scaled: np.ndarray
    imputer: SimpleImputer
    scaler: Any


def prepare_feature_matrix(
    features_df: pd.DataFrame,
    *,
    geo_col: str = "geo_code",
    scaler: str = "robust",
    add_missingness_flags: bool = True,
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

    # Optionally add was_missing flags (helps clustering separate "sparse vs dense" patterns)
    if add_missingness_flags:
        miss_flags = X_df.isna().astype(int)
        miss_flags.columns = [f"{c}__was_missing" for c in miss_flags.columns]
        X_df = pd.concat([X_df, miss_flags], axis=1)

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_df.to_numpy())

    # Drop near-constant / zero-variance features post-imputation (stabilizes GMM especially)
    if drop_zero_variance:
        variances = np.nanvar(X, axis=0)
        keep = variances > 0.0
        X = X[:, keep]
        feature_names = np.array(X_df.columns.astype(str).tolist())[keep].tolist()
    else:
        feature_names = X_df.columns.astype(str).tolist()

    if scaler == "standard":
        scaler_obj = StandardScaler()
    else:
        scaler_obj = RobustScaler(with_centering=True, with_scaling=True)

    X_scaled = scaler_obj.fit_transform(X)
    return PreparedMatrix(
        geo_codes=geo_codes,
        feature_names=feature_names,
        X=X,
        X_scaled=X_scaled,
        imputer=imputer,
        scaler=scaler_obj,
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


