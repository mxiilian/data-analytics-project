from __future__ import annotations

import sys
import logging
import pickle
from pathlib import Path

# Allow running as a script: `python src/clustering/main.py`
# (when executed directly, relative imports have no package context).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.clustering import db as dbmod  # noqa: E402
from src.clustering.models import prepare_feature_matrix, fit_gmm, fit_kmeans  # noqa: E402
from src.clustering.plot import save_cluster_scatter_png  # noqa: E402
from src.clustering.tuning import make_sqlite_storage_uri, tune_clustering  # noqa: E402
from src.clustering.writeback import save_run_results  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CONFIG = {
    # Clustering constraints
    "n_clusters_min": 4,
    "n_clusters_max": 8,
    "min_cluster_size": 1,
    "max_cluster_frac": 0.85,
    
    # Optuna settings
    "optuna_sqlite_path": str(dbmod.default_project_root() / "output" / "optuna" / "optuna_studies.db"),
    "optuna_n_trials": 100,
    "optuna_bootstrap": 100,
    
    "seed": 42,
    "scaler": "robust",  # robust|standard
    "algorithms": ["kmeans", "gmm"],
    
    "export_path": str(dbmod.default_project_root() / "output" / "clustering"),
    "input_feature_csv": str(dbmod.default_project_root() / "output" / "clustering" / "custom_features_notebook.csv"),
}


def _cluster_feature_profiles(features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    X = features_df.set_index("geo_code")
    X = X.apply(pd.to_numeric, errors="coerce")
    global_mean = X.mean(axis=0)
    global_std = X.std(axis=0).replace(0, 1.0)
    out_rows = []
    for cid, idx in labels.groupby(labels).groups.items():
        sub = X.loc[idx]
        mean = sub.mean(axis=0)
        median = sub.median(axis=0)
        z = (mean - global_mean) / global_std
        for feat in X.columns:
            out_rows.append(
                {
                    "cluster_id": int(cid),
                    "feature_name": str(feat),
                    "mean": float(mean[feat]) if pd.notna(mean[feat]) else None,
                    "median": float(median[feat]) if pd.notna(median[feat]) else None,
                    "z_mean": float(z[feat]) if pd.notna(z[feat]) else None,
                }
            )
    return pd.DataFrame(out_rows)


def main() -> None:
    cfg = CONFIG
    seed = int(cfg["seed"])

    optuna_sqlite = Path(cfg["optuna_sqlite_path"])
    optuna_sqlite.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = make_sqlite_storage_uri(optuna_sqlite)

    base_out_dir = Path(cfg["export_path"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Feature Matrix ---
    input_csv = cfg.get("input_feature_csv")
    if not input_csv:
        logger.error("input_feature_csv is required in CONFIG")
        return

    csv_path = Path(input_csv)
    if not csv_path.is_absolute():
        csv_path = dbmod.default_project_root() / csv_path
        
    logger.info("Loading feature matrix from CSV: %s", csv_path)
    if not csv_path.exists():
        logger.error("Input CSV not found: %s", csv_path)
        return
    features_df = pd.read_csv(csv_path)
    
    if features_df.empty or features_df.shape[1] < 2:
        logger.error("Feature matrix is empty or has no features. Exiting.")
        return

    # --- 2. Preprocessing (Scaling / Imputation) ---
    pm = prepare_feature_matrix(
        features_df,
        geo_col="geo_code",
        scaler=str(cfg["scaler"]),
        drop_missing_rate_gt=0.5,
        drop_zero_variance=True,
    )
    
    # Save prepared matrix for inspection
    run_name = f"custom_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = base_out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(pm.X_scaled, columns=pm.feature_names).to_csv(run_dir / "X_scaled.csv", index=False)
    
    # 2D Embedding for plotting
    emb_df = None
    if pm.X_scaled.shape[0] >= 3:
        pca = PCA(n_components=2, random_state=seed)
        emb = pca.fit_transform(pm.X_scaled)
        emb_df = pd.DataFrame({"geo_code": pm.geo_codes, "emb_x": emb[:, 0], "emb_y": emb[:, 1]})

    # --- 3. Clustering & Tuning ---
    for algorithm in cfg["algorithms"]:
        algo = str(algorithm).lower()
        study_name = f"{algo}:{run_name}"
        logger.info("Tuning %s (study=%s)...", algo, study_name)
        
        tuning_res, best_metrics = tune_clustering(
            X_full=pm.X_scaled,
            algorithm=algo,
            study_name=study_name,
            storage_uri=storage_uri,
            n_trials=int(cfg["optuna_n_trials"]),
            seed=seed,
            n_bootstrap=int(cfg["optuna_bootstrap"]),
            min_cluster_size=int(cfg["min_cluster_size"]),
            max_cluster_frac=float(cfg["max_cluster_frac"]),
            n_clusters_min=int(cfg["n_clusters_min"]),
            n_clusters_max=int(cfg["n_clusters_max"]),
        )
        
        logger.info("Best %s params: %s", algo, tuning_res.best_params)
        
        # --- 4. Final Fit ---
        if algo == "kmeans":
            model_res = fit_kmeans(
                pm.X_scaled,
                k=int(tuning_res.best_params["k"]),
                seed=seed,
                n_init=int(tuning_res.best_params.get("n_init", 20)),
                init=str(tuning_res.best_params.get("init", "k-means++")),
                max_iter=int(tuning_res.best_params.get("max_iter", 300)),
            )
        else:
            model_res = fit_gmm(
                pm.X_scaled,
                n_components=int(tuning_res.best_params["n_components"]),
                seed=seed,
                covariance_type=str(tuning_res.best_params.get("covariance_type", "full")),
                reg_covar=float(tuning_res.best_params.get("reg_covar", 1e-6)),
            )
            
        # --- 5. Save Results ---
        algo_dir = run_dir / algo
        
        labels_df = pd.DataFrame({
            "geo_code": pm.geo_codes,
            "cluster_id": model_res.labels,
            "membership_prob": model_res.membership_prob,
        })
        
        profiles_df = _cluster_feature_profiles(features_df, pd.Series(model_res.labels, index=pm.geo_codes))
        
        save_run_results(algo_dir, labels_df, profiles_df)
        
        # Save model
        with open(algo_dir / "model.pkl", "wb") as f:
            pickle.dump(model_res.model, f)
            
        # Save plot
        if emb_df is not None:
            save_cluster_scatter_png(
                emb_df=emb_df,
                labels_df=labels_df,
                title=f"{algo} (k={best_metrics.n_clusters})",
                out_path=algo_dir / "plot.png",
            )
            
    logger.info("Done. Results saved to %s", run_dir)


if __name__ == "__main__":
    main()
