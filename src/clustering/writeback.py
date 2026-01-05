from __future__ import annotations

from typing import Any, Optional

import duckdb

from .db import to_json, utc_now_iso


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    # Keep tables in main schema for simplicity; prefix names.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_runs (
            run_id VARCHAR PRIMARY KEY,
            created_at_utc VARCHAR,
            cluster_type VARCHAR,
            algorithm VARCHAR,
            geo_level VARCHAR,
            year_min INTEGER,
            year_max INTEGER,
            feature_spec_json VARCHAR,
            model_spec_json VARCHAR,
            optuna_storage_uri VARCHAR,
            optuna_study_name VARCHAR,
            best_trial_number INTEGER,
            best_value DOUBLE,
            best_params_json VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_labels (
            run_id VARCHAR,
            cluster_type VARCHAR,
            algorithm VARCHAR,
            geo_code VARCHAR,
            cluster_id INTEGER,
            membership_prob DOUBLE,
            is_noise BOOLEAN,
            PRIMARY KEY (run_id, geo_code)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_metrics (
            run_id VARCHAR,
            metric_name VARCHAR,
            metric_value DOUBLE,
            PRIMARY KEY (run_id, metric_name)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_feature_profiles (
            run_id VARCHAR,
            cluster_id INTEGER,
            feature_name VARCHAR,
            mean DOUBLE,
            median DOUBLE,
            z_mean DOUBLE,
            PRIMARY KEY (run_id, cluster_id, feature_name)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_embeddings (
            run_id VARCHAR,
            geo_code VARCHAR,
            method VARCHAR,
            emb_x DOUBLE,
            emb_y DOUBLE,
            PRIMARY KEY (run_id, geo_code, method)
        );
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW v_cluster_labels_latest AS
        SELECT l.*
        FROM cluster_labels l
        JOIN (
            SELECT
                cluster_type,
                arg_max(run_id, created_at_utc) AS run_id
            FROM cluster_runs
            GROUP BY cluster_type
        ) latest
        ON latest.run_id = l.run_id
        AND latest.cluster_type = l.cluster_type;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW v_cluster_labels_latest_by_algo AS
        SELECT l.*
        FROM cluster_labels l
        JOIN (
            SELECT
                cluster_type,
                algorithm,
                arg_max(run_id, created_at_utc) AS run_id
            FROM cluster_runs
            GROUP BY cluster_type, algorithm
        ) latest
        ON latest.run_id = l.run_id
        AND latest.cluster_type = l.cluster_type
        AND latest.algorithm = l.algorithm;
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW v_cluster_runs_latest AS
        SELECT r.*
        FROM cluster_runs r
        JOIN (
            SELECT
                cluster_type,
                algorithm,
                arg_max(run_id, created_at_utc) AS run_id
            FROM cluster_runs
            GROUP BY cluster_type, algorithm
        ) latest
        ON latest.run_id = r.run_id
        AND latest.cluster_type = r.cluster_type
        AND latest.algorithm = r.algorithm;
        """
    )


def write_run(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    cluster_type: str,
    algorithm: str,
    geo_level: str,
    year_min: int,
    year_max: int,
    feature_spec: dict[str, Any],
    model_spec: dict[str, Any],
    optuna_storage_uri: Optional[str],
    optuna_study_name: Optional[str],
    best_trial_number: Optional[int],
    best_value: Optional[float],
    best_params: Optional[dict[str, Any]],
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO cluster_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            cluster_type,
            algorithm,
            geo_level,
            int(year_min),
            int(year_max),
            to_json(feature_spec),
            to_json(model_spec),
            optuna_storage_uri,
            optuna_study_name,
            best_trial_number,
            best_value,
            to_json(best_params) if best_params is not None else None,
        ),
    )


def write_metrics(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    metrics: dict[str, float],
) -> None:
    rows = [(run_id, k, float(v)) for k, v in metrics.items()]
    con.execute("DELETE FROM cluster_metrics WHERE run_id = ?", (run_id,))
    con.executemany(
        "INSERT INTO cluster_metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?)",
        rows,
    )


def write_labels_df(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    cluster_type: str,
    algorithm: str,
    labels_df,
) -> None:
    # labels_df columns: geo_code, cluster_id, membership_prob, is_noise
    labels_df = labels_df.copy()
    labels_df["run_id"] = run_id
    labels_df["cluster_type"] = cluster_type
    labels_df["algorithm"] = algorithm
    con.register("tmp_cluster_labels", labels_df)
    con.execute("DELETE FROM cluster_labels WHERE run_id = ?", (run_id,))
    con.execute(
        """
        INSERT INTO cluster_labels
        SELECT run_id, cluster_type, algorithm, geo_code, cluster_id, membership_prob, is_noise
        FROM tmp_cluster_labels
        """
    )
    con.unregister("tmp_cluster_labels")


def write_feature_profiles_df(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    profiles_df,
) -> None:
    # profiles_df columns: cluster_id, feature_name, mean, median, z_mean
    profiles_df = profiles_df.copy()
    profiles_df["run_id"] = run_id
    con.register("tmp_cluster_profiles", profiles_df)
    con.execute("DELETE FROM cluster_feature_profiles WHERE run_id = ?", (run_id,))
    con.execute(
        """
        INSERT INTO cluster_feature_profiles
        SELECT run_id, cluster_id, feature_name, mean, median, z_mean
        FROM tmp_cluster_profiles
        """
    )
    con.unregister("tmp_cluster_profiles")


def write_embeddings_df(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    method: str,
    emb_df,
) -> None:
    # emb_df columns: geo_code, emb_x, emb_y
    emb_df = emb_df.copy()
    emb_df["run_id"] = run_id
    emb_df["method"] = method
    con.register("tmp_cluster_emb", emb_df)
    con.execute("DELETE FROM cluster_embeddings WHERE run_id = ?", (run_id,))
    con.execute(
        """
        INSERT INTO cluster_embeddings
        SELECT run_id, geo_code, method, emb_x, emb_y
        FROM tmp_cluster_emb
        """
    )
    con.unregister("tmp_cluster_emb")


