from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd


def _norm_city_name(s: str) -> str:
    s = (s or "").strip().lower()
    # Take only the city part if geo_name is like "City, Country"
    s = s.split(",")[0]
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _who_city_part_from_geo_code(who_geo_code: str) -> str:
    # "DEU_Hamburg" -> "Hamburg"
    if not who_geo_code or "_" not in who_geo_code:
        return who_geo_code
    return who_geo_code.split("_", 1)[1]


def build_who_city_map(
    eurostat_cities: pd.DataFrame,
    who_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a simple mapping from Eurostat cities (geo_code/geo_name) to WHO geo_code.

    Strategy (simple on purpose):
    - Normalize Eurostat geo_name city part
    - Normalize WHO city part from WHO geo_code (after ISO3_) and WHO geo_name (before comma)
    - Match by normalized city name

    Returns:
    - eurostat_geo_code, eurostat_geo_name, who_geo_code, who_geo_name
    """
    if eurostat_cities.empty or who_long.empty:
        return pd.DataFrame(columns=["eurostat_geo_code", "eurostat_geo_name", "who_geo_code", "who_geo_name"])

    who_geo = who_long[["who_geo_code", "who_geo_name"]].drop_duplicates().copy()
    who_geo["who_city_norm"] = who_geo["who_geo_code"].map(_who_city_part_from_geo_code).map(_norm_city_name)
    who_geo["who_name_norm"] = who_geo["who_geo_name"].map(_norm_city_name)

    euro = eurostat_cities[["geo_code", "geo_name"]].drop_duplicates().copy()
    euro["euro_city_norm"] = euro["geo_name"].map(_norm_city_name)

    # Prefer matching against WHO city part; fallback to WHO geo_name.
    m1 = euro.merge(
        who_geo[["who_geo_code", "who_geo_name", "who_city_norm"]],
        left_on="euro_city_norm",
        right_on="who_city_norm",
        how="left",
    )
    missing = m1["who_geo_code"].isna()
    if missing.any():
        m2 = euro[missing].merge(
            who_geo[["who_geo_code", "who_geo_name", "who_name_norm"]],
            left_on="euro_city_norm",
            right_on="who_name_norm",
            how="left",
        )
        m1.loc[missing, ["who_geo_code", "who_geo_name"]] = m2[["who_geo_code", "who_geo_name"]].to_numpy()

    out = m1.rename(
        columns={"geo_code": "eurostat_geo_code", "geo_name": "eurostat_geo_name"}
    )[["eurostat_geo_code", "eurostat_geo_name", "who_geo_code", "who_geo_name"]]
    return out


def left_join_who_as_city_features(
    *,
    eurostat_cities: pd.DataFrame,
    who_long: pd.DataFrame,
    indicator_codes: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left-join WHO air quality onto Eurostat cities (city entities stay Eurostat-only).

    Returns:
    - who_enriched_long: columns [city_geo_code, year, indicator_code, value]
    - mapping_df: eurostat→who mapping used (for transparency)
    """
    mapping = build_who_city_map(eurostat_cities, who_long)
    mapped = mapping.dropna(subset=["who_geo_code"]).copy()

    if mapped.empty:
        return (
            pd.DataFrame(columns=["city_geo_code", "year", "indicator_code", "value"]),
            mapping,
        )

    who = who_long[who_long["indicator_code"].isin(list(indicator_codes))].copy()
    who = who.merge(mapped[["eurostat_geo_code", "who_geo_code"]], on="who_geo_code", how="inner")

    who_out = who.rename(columns={"eurostat_geo_code": "city_geo_code"})[
        ["city_geo_code", "year", "indicator_code", "value"]
    ].copy()

    return who_out, mapping

