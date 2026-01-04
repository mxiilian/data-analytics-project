from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import duckdb
import pandas as pd


CITY_CODE_REGEX = r"^[A-Z]{3}_"  # WHO geo_code format from scripts/database-pipeline.py


@dataclass(frozen=True)
class CityEntity:
    geo_code: str
    geo_name: str
    country_code: Optional[str]


def fetch_city_entities(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: geo_code, geo_name, country_code.

    City selection:
    - Eurostat city codes: ends with C or K
    - WHO geo codes: ISO3_CityName (regex ^[A-Z]{3}_)
    - Excludes ISO2 countries (length 2)
    """
    return con.execute(
        f"""
        SELECT
            geo_code,
            geo_name,
            country_code
        FROM dim_geo
        WHERE
            length(geo_code) > 2
            AND (
                right(geo_code, 1) IN ('C', 'K')
                OR regexp_matches(geo_code, '{CITY_CODE_REGEX}')
            )
        ORDER BY geo_code
        """
    ).df()


def fetch_long_city_measurements(
    con: duckdb.DuckDBPyConnection,
    *,
    year_min: int,
    year_max: int,
    indicator_code_whitelist: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Long-form city-level measurements:
    columns: geo_code, year, indicator_code, value, flag
    """
    where_ind = ""
    if indicator_code_whitelist:
        con.execute("CREATE TEMP TABLE IF NOT EXISTS _tmp_ind_whitelist AS SELECT 1 AS x;")
        # Use a values list via a relation register for safety.
        wl = pd.DataFrame({"indicator_code": indicator_code_whitelist})
        con.register("tmp_ind_whitelist", wl)
        where_ind = " AND f.indicator_code IN (SELECT indicator_code FROM tmp_ind_whitelist) "

    df = con.execute(
        f"""
        WITH cities AS (
            SELECT geo_code
            FROM dim_geo
            WHERE
                length(geo_code) > 2
                AND (
                    right(geo_code, 1) IN ('C', 'K')
                    OR regexp_matches(geo_code, '{CITY_CODE_REGEX}')
                )
        )
        SELECT
            f.geo_code,
            f.year,
            f.indicator_code,
            f.value,
            f.flag
        FROM fact_measurements f
        JOIN cities c ON c.geo_code = f.geo_code
        WHERE f.year BETWEEN {int(year_min)} AND {int(year_max)}
        {where_ind}
        """
    ).df()

    if indicator_code_whitelist:
        con.unregister("tmp_ind_whitelist")

    return df


def fetch_long_country_measurements_for_cities(
    con: duckdb.DuckDBPyConnection,
    *,
    year_min: int,
    year_max: int,
    indicator_code_whitelist: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Country-level measurements joined onto each city by ISO2 country_code.

    Returns long-form enriched rows with columns:
    - city_geo_code, year, indicator_code, value

    Notes:
    - Only enriches for cities whose `dim_geo.country_code` is ISO2 (length 2).
      This covers Eurostat city codes (e.g., DE002C -> DE).
    - WHO city rows typically have ISO3 country_code and will not enrich.
    - Output is intended to be merged into the city feature set; it does not change entity grain.
    """
    where_ind = ""
    if indicator_code_whitelist:
        wl = pd.DataFrame({"indicator_code": indicator_code_whitelist})
        con.register("tmp_ind_whitelist2", wl)
        where_ind = " AND f2.indicator_code IN (SELECT indicator_code FROM tmp_ind_whitelist2) "

    df = con.execute(
        f"""
        WITH cities AS (
            SELECT
                geo_code AS city_geo_code,
                country_code
            FROM dim_geo
            WHERE
                length(geo_code) > 2
                AND (
                    right(geo_code, 1) IN ('C', 'K')
                    OR regexp_matches(geo_code, '{CITY_CODE_REGEX}')
                )
                AND length(country_code) = 2
        )
        SELECT
            c.city_geo_code,
            f2.year,
            f2.indicator_code,
            f2.value
        FROM cities c
        JOIN fact_measurements f2
            ON f2.geo_code = c.country_code
        WHERE f2.year BETWEEN {int(year_min)} AND {int(year_max)}
        {where_ind}
        """
    ).df()

    if indicator_code_whitelist:
        con.unregister("tmp_ind_whitelist2")

    return df


