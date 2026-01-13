# Clustering smoke checklist (DuckDB + Optuna)

This repository implements clustering in `src/clustering/main.py`:
- Reads cities from `data/db/data-full.duckdb`
- Enriches city features with country indicators (ISO2 join)
- Tunes KMeans/GMM via Optuna using SQLite storage
- Writes results back into DuckDB tables

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run clustering

```bash
python3 -m src.clustering.main
```

Expected side effects:
- Optuna study DB created/updated at `output/optuna/optuna_studies.db`
- DuckDB tables created/updated in `data/db/data-full.duckdb`:
  - `cluster_runs`, `cluster_labels`, `cluster_metrics`, `cluster_feature_profiles`, `cluster_embeddings`

## 3) Quick DB checks

Open DuckDB (or query via Python) and run:

```sql
SELECT * FROM cluster_runs ORDER BY created_at_utc DESC LIMIT 5;
SELECT cluster_type, algorithm, COUNT(*) AS n_cities
FROM cluster_labels
GROUP BY 1,2
ORDER BY 1,2;
SELECT * FROM cluster_metrics LIMIT 20;
```

## 4) Resume check (Optuna SQLite)

Re-run `python3 -m src.clustering.main`.\n
Expected:
- Studies load existing history (same `study_name`) and append more trials.
- New `cluster_runs` rows created, pointing to the same `optuna_storage_uri` + `optuna_study_name`.



