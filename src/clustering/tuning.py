from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .evaluate import ClusterMetrics, compute_cluster_metrics, objective_score
from .models import fit_gmm, fit_kmeans


@dataclass(frozen=True)
class TuningResult:
    study_name: str
    storage_uri: str
    best_trial_number: int
    best_value: float
    best_params: dict[str, Any]


def make_sqlite_storage_uri(sqlite_path: Path) -> str:
    # Optuna expects SQLAlchemy URI; for absolute file path: sqlite:////abs/path.db
    p = sqlite_path.resolve()
    return f"sqlite:////{str(p).lstrip('/')}"


def create_or_load_study(
    *,
    study_name: str,
    storage_uri: str,
    direction: str = "maximize",
    seed: int = 42,
    use_pruner: bool = True,
) -> optuna.Study:
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner() if use_pruner else optuna.pruners.NopPruner()
    return optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _fit_predict_factory(
    algorithm: str,
    params: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], Any]:
    algo = algorithm.lower()
    if algo == "kmeans":
        res = fit_kmeans(
            params["_X_full"],
            k=int(params["k"]),
            seed=seed,
            n_init=int(params.get("n_init", 20)),
            init=str(params.get("init", "k-means++")),
            max_iter=int(params.get("max_iter", 300)),
        )
        return res.labels, res.membership_prob, res.model
    if algo == "gmm":
        res = fit_gmm(
            params["_X_full"],
            n_components=int(params["n_components"]),
            seed=seed,
            covariance_type=str(params.get("covariance_type", "full")),
            reg_covar=float(params.get("reg_covar", 1e-6)),
        )
        return res.labels, res.membership_prob, res.model
    raise ValueError(f"Unknown algorithm: {algorithm}")


def tune_clustering(
    *,
    X_full: np.ndarray,
    algorithm: str,
    study_name: str,
    storage_uri: str,
    n_trials: int,
    seed: int = 42,
    n_bootstrap: int = 100,
    min_cluster_size: int = 2,
    max_cluster_frac: float = 0.85,
    use_pruner: bool = True,
) -> tuple[TuningResult, ClusterMetrics]:
    """
    Runs Optuna tuning and returns best params + best metrics (computed on full data).
    """
    X_full = np.asarray(X_full)
    n = X_full.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 cities to cluster.")

    study = create_or_load_study(
        study_name=study_name,
        storage_uri=storage_uri,
        seed=seed,
        use_pruner=use_pruner,
    )

    algo = algorithm.lower()

    def objective(trial: optuna.Trial) -> float:
        # Suggest params
        if algo == "kmeans":
            k_max = max(2, min(12, n - 1))
            k = trial.suggest_int("k", 2, k_max)
            n_init = trial.suggest_int("n_init", 10, 50)
            init = trial.suggest_categorical("init", ["k-means++", "random"])
            max_iter = trial.suggest_int("max_iter", 100, 600)
            params = {"k": k, "n_init": n_init, "init": init, "max_iter": max_iter, "_X_full": X_full}
        elif algo == "gmm":
            k_max = max(2, min(12, n - 1))
            n_components = trial.suggest_int("n_components", 2, k_max)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "diag", "tied", "spherical"])
            # Wider regularization range for numerical stability on small-N/high-dim data.
            reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-2, log=True)
            params = {
                "n_components": n_components,
                "covariance_type": covariance_type,
                "reg_covar": reg_covar,
                "_X_full": X_full,
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Fit on full data (GMM can fail for some hyperparams / degenerate covariance)
        try:
            labels, _, _model = _fit_predict_factory(algorithm, params, seed=seed)
        except Exception as e:
            # Treat as a bad trial, but do not fail the optimization run.
            trial.set_user_attr("fit_failed", True)
            trial.set_user_attr("fit_error", repr(e))
            return -1e9

        def fit_predict_boot(X_train: np.ndarray, X_predict: np.ndarray) -> np.ndarray:
            # Fit a fresh model on X_train; then predict labels for X_predict.
            if algo == "kmeans":
                km = fit_kmeans(
                    X_train,
                    k=int(params.get("k", params.get("n_components"))),
                    seed=seed,
                    n_init=int(params.get("n_init", 20)),
                    init=str(params.get("init", "k-means++")),
                    max_iter=int(params.get("max_iter", 300)),
                ).model
                return km.predict(X_predict)
            gm = fit_gmm(
                X_train,
                n_components=int(params.get("n_components")),
                seed=seed,
                covariance_type=str(params.get("covariance_type", "full")),
                reg_covar=float(params.get("reg_covar", 1e-6)),
            ).model
            return gm.predict(X_predict)

        metrics = compute_cluster_metrics(
            X=X_full,
            labels=labels,
            fit_predict_full=fit_predict_boot,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        score = objective_score(metrics, min_cluster_size=min_cluster_size, max_cluster_frac=max_cluster_frac)

        # Track metadata inside Optuna RDB
        trial.set_user_attr(
            "metrics",
            {
                "silhouette": metrics.silhouette,
                "davies_bouldin": metrics.davies_bouldin,
                "calinski_harabasz": metrics.calinski_harabasz,
                "stability_ari": metrics.stability_ari,
                "n_clusters": metrics.n_clusters,
                "min_cluster_size": metrics.min_cluster_size,
                "max_cluster_frac": metrics.max_cluster_frac,
            },
        )
        trial.set_user_attr(
            "constraints",
            {
                "min_cluster_size": min_cluster_size,
                "max_cluster_frac": max_cluster_frac,
            },
        )

        # Single-step report (pruning generally not meaningful here)
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(score)

    start = time.time()
    study.optimize(objective, n_trials=int(n_trials))
    _ = time.time() - start

    # If all trials were invalid and got the same sentinel score, Optuna may still pick one.
    # Ensure we have at least one completed trial with a non-sentinel value.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(
            f"Optuna produced no completed trials for algorithm={algorithm}. "
            f"Consider disabling it or widening reg_covar / reducing k."
        )

    best = study.best_trial
    best_params = dict(best.params)
    # Recompute best metrics on full data
    params2 = dict(best_params)
    params2["_X_full"] = X_full
    labels, _, _ = _fit_predict_factory(algorithm, params2, seed=seed)

    def fit_predict_boot_best(X_train: np.ndarray, X_predict: np.ndarray) -> np.ndarray:
        if algo == "kmeans":
            km = fit_kmeans(
                X_train,
                k=int(best_params["k"]),
                seed=seed,
                n_init=int(best_params.get("n_init", 20)),
                init=str(best_params.get("init", "k-means++")),
                max_iter=int(best_params.get("max_iter", 300)),
            ).model
            return km.predict(X_predict)
        gm = fit_gmm(
            X_train,
            n_components=int(best_params["n_components"]),
            seed=seed,
            covariance_type=str(best_params.get("covariance_type", "full")),
            reg_covar=float(best_params.get("reg_covar", 1e-6)),
        ).model
        return gm.predict(X_predict)

    best_metrics = compute_cluster_metrics(
        X=X_full,
        labels=labels,
        fit_predict_full=fit_predict_boot_best,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    return (
        TuningResult(
            study_name=study.study_name,
            storage_uri=storage_uri,
            best_trial_number=int(best.number),
            best_value=float(study.best_value),
            best_params=best_params,
        ),
        best_metrics,
    )


