from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    db_path: Path
    year_min: int
    year_max: int
    created_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_project_root() -> Path:
    # src/clustering/db.py -> project root is 3 levels up
    return Path(__file__).resolve().parents[2]


def default_data_full_db_path() -> Path:
    return default_project_root() / "data" / "db" / "data-full.duckdb"


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


def make_run_id(prefix: str = "cluster") -> str:
    # Deterministic enough for local runs; includes ms timestamp.
    return f"{prefix}_{int(time.time() * 1000)}"


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


