# Clustering (DuckDB + Optuna)

This package implements a **city-level clustering pipeline** on top of the project DuckDB star schema (`data/db/data-full.duckdb`).\n
It builds three types of feature matrices (quality / typology / trajectory), tunes clustering hyperparameters with **Optuna** using **SQLite storage**, and writes results back into DuckDB (plus CSV/PNG exports).

## How to run

From project root:

```bash
python src/clustering/main.py
```

Alternative (module mode):

```bash
python -m src.clustering.main
```

## What data is used

### Source tables (DuckDB)

The pipeline reads:
- **`dim_geo`**: used to identify city entities and country mapping
- **`fact_measurements`**: the long-form measurement table (`geo_code`, `year`, `indicator_code`, `value`, `flag`)

### City entity list (cities only)

Cities are identified in `src/clustering/extract.py` as:
- **Eurostat cities**: `geo_code` ends with `C` or `K`
- **WHO cities**: `geo_code` matches `^[A-Z]{3}_` (ISO3 prefix used by `scripts/database-pipeline.py`)
- **Countries excluded**: ISO2 geos (length 2) are excluded from the clustering entity list

### Country-as-features enrichment

Country-level measurements are joined onto each **city-year** row by:
- `city.dim_geo.country_code` (ISO2) == `fact_measurements.geo_code` (ISO2 country)

Notes:
- This primarily enriches **Eurostat city codes** (e.g. `DE002C` → `DE`).\n
- WHO city geos often have ISO3 `country_code` and therefore **do not** enrich automatically.

## Feature matrices (what the model actually sees)

All feature matrices are computed per **city** (one row per `geo_code`).\n
They are built from **city measurements + enriched country measurements** in the configured year range (defaults: 2015–2023).

### 1) Quality (availability / data quality) clusters

Built by `build_quality_features()` in `src/clustering/features.py`.

Per city, we compute:
- `n_indicators_with_any_data`
- `mean_avail_ratio`, `median_avail_ratio`
- `pct_indicators_ge_0_7`, `pct_indicators_ge_0_9`
- `flag_rate` (fraction of rows with a non-empty `flag`)
- `avail_{indicator_code}` for a top-N set of indicators (default: 30), selected by high cross-city variability in availability ratios

Availability ratios are computed as:
- observed years for `(city, indicator)` divided by **expected years for that indicator**\n
  (expected years = number of distinct years that indicator appears in the dataset).\n
This avoids over-penalizing indicators that simply don’t exist in early years.

### 2) Typology (multi-year summary) clusters

Built by `build_typology_features()` in `src/clustering/features.py`.

For each `(city, indicator)`, compute:
- `mean`\n
- `last` (last available value in range)\n
- `slope` (linear regression slope of `value ~ year`, only if ≥3 points)\n
- `volatility` (std of YoY % change)

Then pivot wide to one row per city with columns like:
- `{indicator_code}__mean`
- `{indicator_code}__last`
- `{indicator_code}__slope`
- `{indicator_code}__volatility`

### 3) Trajectory (development pattern) clusters

Built by `build_trajectory_features()` in `src/clustering/features.py`.

For each `(city, indicator)`, compute:
- `slope`\n
- `delta` (last − first)\n
- `volatility` (std of YoY % change)

Pivot wide to columns like:
- `{indicator_code}__traj_slope`
- `{indicator_code}__traj_delta`
- `{indicator_code}__traj_volatility`

## Missing data handling + feature filtering

Feature preparation is done in `prepare_feature_matrix()` in `src/clustering/models.py`:

1) **Drop near-empty columns**\n
   - any feature with missing rate > `drop_missing_rate_gt` (default **0.98**) is removed

2) **Missingness flags (recommended)**\n
   - for every remaining feature, add a binary column `{feature}__was_missing` (0/1)

3) **Imputation**\n
   - `SimpleImputer(strategy="median")`

4) **Drop zero-variance columns**\n
   - removes features that became constant after imputation (important for GMM stability)

5) **Scaling**\n
   - default: `RobustScaler`\n
   - optional: `StandardScaler`

## Algorithms + hyperparameter tuning (Optuna)

Tuning lives in `src/clustering/tuning.py` and runs per `(cluster_type, algorithm)`.

### Optuna storage (SQLite)

- Storage file (default): `output/optuna/optuna_studies.db`\n
- Storage URI: `sqlite:////ABS/PATH/output/optuna/optuna_studies.db`\n
- Studies are created with `load_if_exists=True` so tuning **resumes** across runs.\n
This follows Optuna’s RDB backend approach: [Saving/Resuming Study with RDB Backend](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html).

Study names are **stable** and include:\n
`{cluster_type}:{algorithm}:{year_min}-{year_max}:{scaler}`

### KMeans

Model: `sklearn.cluster.KMeans`

Tuned parameters:
- `k`: 2 .. min(12, n_cities-1)\n
- `n_init`: 10 .. 50\n
- `init`: `k-means++` | `random`\n
- `max_iter`: 100 .. 600

### GMM (Gaussian Mixture)

Model: `sklearn.mixture.GaussianMixture`

Tuned parameters:
- `n_components`: 2 .. min(12, n_cities-1)\n
- `covariance_type`: `full` | `diag` | `tied` | `spherical`\n
- `reg_covar`: 1e-6 .. 1e-2 (log scale)

Notes:
- GMM can fail to fit for some hyperparams on small datasets (ill-defined covariances).\n
  These trials are **recorded** in Optuna with `fit_failed=True` and are scored as very poor rather than crashing the run.

### Objective + metrics

For each trial, we compute:
- silhouette\n
- Davies–Bouldin\n
- Calinski–Harabasz\n
- stability_ari (bootstrap ARI vs baseline clustering)\n
- min cluster size + max cluster fraction

Single scalar objective (maximize):

\[
objective = silhouette - 0.2 \\cdot davies\\_bouldin + 0.5 \\cdot stability\\_ari
\]

Hard constraints (invalid → very low score):
- < 2 clusters\n
- `min_cluster_size` too small\n
- `max_cluster_frac` too large (dominant single cluster)

## Outputs

### DuckDB tables (written)

All results are written into `data/db/data-full.duckdb`:\n
- `cluster_runs`: one row per saved clustering run, including Optuna study pointers and best params\n
- `cluster_labels`: `geo_code → cluster_id` (+ `membership_prob` for GMM)\n
- `cluster_metrics`: metrics for the saved best trial\n
- `cluster_feature_profiles`: interpretability table (per cluster mean/median and z-score vs global)\n
- `cluster_embeddings`: stored 2D PCA embedding coordinates (`method='pca2'`)

Views:
- `v_cluster_labels_latest`\n
- `v_cluster_labels_latest_by_algo`\n
- `v_cluster_runs_latest`

### Exported artifacts (files)

Feature matrices (exact post-imputation matrices used for clustering):
- `output/clustering/feature_matrices/{cluster_type}__X.csv`\n
- `output/clustering/feature_matrices/{cluster_type}__X_scaled_{scaler}.csv`

Plots (if matplotlib is installed):
- `output/clustering/plots/{cluster_type}__{algo}__{run_id}.png`

## Where configuration lives

All runtime configuration is a dict in:\n
- `src/clustering/main.py` (`CONFIG`)

This includes year range, scaler choice, missingness handling toggles, Optuna trial count, bootstrap count, and enabled algorithms.

