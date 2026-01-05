from __future__ import annotations

import sys
import logging
from pathlib import Path

# Allow running as a script: `python src/clustering/main.py`
# (when executed directly, relative imports have no package context).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from sklearn.decomposition import PCA

from src.clustering import db as dbmod
from src.clustering.evaluate import ClusterMetrics
from src.clustering.extract import (
    fetch_city_entities,
    fetch_long_city_measurements,
    fetch_long_country_measurements_for_cities,
)
from src.clustering.features import build_quality_features, build_trajectory_features, build_typology_features
from src.clustering.models import prepare_feature_matrix, fit_gmm, fit_kmeans
from src.clustering.tuning import make_sqlite_storage_uri, tune_clustering
from src.clustering.writeback import (
    ensure_schema,
    write_embeddings_df,
    write_feature_profiles_df,
    write_labels_df,
    write_metrics,
    write_run,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CONFIG = {
    "db_path": str(dbmod.default_data_full_db_path()),
    "year_min": 2015,
    "year_max": 2023,
    "geo_level": "city",
    # Optuna SQLite storage file
    "optuna_sqlite_path": str(dbmod.default_project_root() / "output" / "optuna" / "optuna_studies.db"),
    "optuna_n_trials": 80,
    "optuna_bootstrap": 100,
    "min_cluster_size": 2,
    "max_cluster_frac": 0.85,
    "seed": 42,
    "scaler": "robust",  # robust|standard
    "add_missingness_flags": True,
    "drop_missing_rate_gt": 0.98,
    "drop_zero_variance": True,
    "export_feature_matrices": True,
    "run_quality": True,
    "run_typology": True,
    "run_trajectory": True,
    "algorithms": ["kmeans", "gmm"],
    "pca_embedding": True,
}


def _cluster_feature_profiles(features_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    # features_df includes geo_code + feature columns
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


def _labels_df(geo_codes: list[str], labels, membership_prob=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "geo_code": geo_codes,
            "cluster_id": labels.astype(int),
            "membership_prob": membership_prob if membership_prob is not None else None,
            "is_noise": False,
        }
    )
    return df


def _metrics_dict(m: ClusterMetrics) -> dict[str, float]:
    return {
        "silhouette": m.silhouette,
        "davies_bouldin": m.davies_bouldin,
        "calinski_harabasz": m.calinski_harabasz,
        "stability_ari": m.stability_ari,
        "n_clusters": float(m.n_clusters),
        "min_cluster_size": float(m.min_cluster_size),
        "max_cluster_frac": float(m.max_cluster_frac),
    }


def main() -> None:
    cfg = CONFIG
    db_path = Path(cfg["db_path"])
    year_min = int(cfg["year_min"])
    year_max = int(cfg["year_max"])
    seed = int(cfg["seed"])

    optuna_sqlite = Path(cfg["optuna_sqlite_path"])
    optuna_sqlite.parent.mkdir(parents=True, exist_ok=True)
    storage_uri = make_sqlite_storage_uri(optuna_sqlite)

    feature_out_dir = dbmod.default_project_root() / "output" / "clustering" / "feature_matrices"
    if cfg.get("export_feature_matrices", True):
        feature_out_dir.mkdir(parents=True, exist_ok=True)

    con = dbmod.connect(db_path)
    ensure_schema(con)

    logger.info("Fetching city entities…")
    cities = fetch_city_entities(con)
    logger.info("Cities found: %d", len(cities))

    logger.info("Extracting long measurements…")
    city_long = fetch_long_city_measurements(con, year_min=year_min, year_max=year_max)
    country_long = fetch_long_country_measurements_for_cities(con, year_min=year_min, year_max=year_max)
    logger.info("City rows: %d | Country-enriched rows: %d", len(city_long), len(country_long))

    cluster_jobs: list[tuple[str, pd.DataFrame]] = []
    if cfg["run_quality"]:
        cluster_jobs.append(
            (
                "quality",
                build_quality_features(
                    city_long=city_long,
                    country_enriched_long=country_long,
                    year_min=year_min,
                    year_max=year_max,
                ),
            )
        )
    if cfg["run_typology"]:
        cluster_jobs.append(
            (
                "typology",
                build_typology_features(
                    city_long=city_long,
                    country_enriched_long=country_long,
                    year_min=year_min,
                    year_max=year_max,
                ),
            )
        )
    if cfg["run_trajectory"]:
        cluster_jobs.append(
            (
                "trajectory",
                build_trajectory_features(
                    city_long=city_long,
                    country_enriched_long=country_long,
                    year_min=year_min,
                    year_max=year_max,
                ),
            )
        )

    for cluster_type, features_df in cluster_jobs:
        logger.info("=== Cluster type: %s ===", cluster_type)
        pm = prepare_feature_matrix(
            features_df,
            geo_col="geo_code",
            scaler=str(cfg["scaler"]),
            add_missingness_flags=bool(cfg["add_missingness_flags"]),
            drop_missing_rate_gt=float(cfg["drop_missing_rate_gt"]),
            drop_zero_variance=bool(cfg["drop_zero_variance"]),
        )

        # Export the exact feature matrices used for clustering (post filtering + imputation).
        if cfg.get("export_feature_matrices", True):
            X_df = pd.DataFrame(pm.X, columns=pm.feature_names)
            X_df.insert(0, "geo_code", pm.geo_codes)
            X_scaled_df = pd.DataFrame(pm.X_scaled, columns=pm.feature_names)
            X_scaled_df.insert(0, "geo_code", pm.geo_codes)

            (feature_out_dir / f"{cluster_type}__X.csv").write_text(
                X_df.to_csv(index=False),
                encoding="utf-8",
            )
            (feature_out_dir / f"{cluster_type}__X_scaled_{cfg['scaler']}.csv").write_text(
                X_scaled_df.to_csv(index=False),
                encoding="utf-8",
            )
            logger.info(
                "Wrote feature matrices: %s (%d cities, %d features)",
                feature_out_dir,
                X_df.shape[0],
                X_df.shape[1] - 1,
            )

        # Optional 2D embedding for plotting (PCA on scaled features)
        emb_df = None
        if cfg["pca_embedding"] and pm.X_scaled.shape[0] >= 3:
            pca = PCA(n_components=2, random_state=seed)
            emb = pca.fit_transform(pm.X_scaled)
            emb_df = pd.DataFrame({"geo_code": pm.geo_codes, "emb_x": emb[:, 0], "emb_y": emb[:, 1]})

        for algorithm in cfg["algorithms"]:
            algo = str(algorithm).lower()
            run_id = dbmod.make_run_id(f"{cluster_type}_{algo}")
            # Stable study name so tuning can be resumed across runs.
            study_name = f"{cluster_type}:{algo}:{year_min}-{year_max}:{cfg['scaler']}"

            logger.info("Tuning %s with Optuna (study=%s)", algo, study_name)
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
                use_pruner=True,
            )

            # Fit final model with best params and compute labels
            if algo == "kmeans":
                fr = fit_kmeans(
                    pm.X_scaled,
                    k=int(tuning_res.best_params["k"]),
                    seed=seed,
                    n_init=int(tuning_res.best_params.get("n_init", 20)),
                    init=str(tuning_res.best_params.get("init", "k-means++")),
                    max_iter=int(tuning_res.best_params.get("max_iter", 300)),
                )
            else:
                fr = fit_gmm(
                    pm.X_scaled,
                    n_components=int(tuning_res.best_params["n_components"]),
                    seed=seed,
                    covariance_type=str(tuning_res.best_params.get("covariance_type", "full")),
                    reg_covar=float(tuning_res.best_params.get("reg_covar", 1e-6)),
                )

            labels_series = pd.Series(fr.labels, index=pm.geo_codes, name="cluster_id")
            labels_df = _labels_df(pm.geo_codes, fr.labels, fr.membership_prob)
            profiles_df = _cluster_feature_profiles(features_df, labels_series)

            feature_spec = {
                "cluster_type": cluster_type,
                "scaler": cfg["scaler"],
                "year_min": year_min,
                "year_max": year_max,
                "country_enrichment": True,
            }
            model_spec = {"algorithm": algo, "best_params": tuning_res.best_params}

            write_run(
                con,
                run_id=run_id,
                cluster_type=cluster_type,
                algorithm=algo,
                geo_level=str(cfg["geo_level"]),
                year_min=year_min,
                year_max=year_max,
                feature_spec=feature_spec,
                model_spec=model_spec,
                optuna_storage_uri=tuning_res.storage_uri,
                optuna_study_name=tuning_res.study_name,
                best_trial_number=tuning_res.best_trial_number,
                best_value=tuning_res.best_value,
                best_params=tuning_res.best_params,
            )
            write_metrics(con, run_id=run_id, metrics=_metrics_dict(best_metrics))
            write_labels_df(con, run_id=run_id, cluster_type=cluster_type, algorithm=algo, labels_df=labels_df)
            write_feature_profiles_df(con, run_id=run_id, profiles_df=profiles_df)
            if emb_df is not None:
                write_embeddings_df(con, run_id=run_id, method="pca2", emb_df=emb_df)

            logger.info(
                "Saved run %s: %s/%s best_value=%.4f clusters=%d",
                run_id,
                cluster_type,
                algo,
                tuning_res.best_value,
                best_metrics.n_clusters,
            )

    con.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()


