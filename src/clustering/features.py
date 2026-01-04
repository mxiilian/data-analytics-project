from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    year_min: int
    year_max: int
    include_country_enrichment: bool = True
    top_n_availability_indicators: int = 30


def _expected_years_per_indicator(long_df: pd.DataFrame, year_min: int, year_max: int) -> pd.Series:
    """
    Expected years = number of distinct years where the indicator appears in the dataset.
    This avoids penalizing indicators that don't exist across the full range.
    """
    df = long_df.dropna(subset=["value"]).copy()
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    return df.groupby("indicator_code")["year"].nunique()


def build_quality_features(
    *,
    city_long: pd.DataFrame,
    country_enriched_long: Optional[pd.DataFrame],
    year_min: int,
    year_max: int,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Returns features indexed by city geo_code.
    """
    # Normalize shapes
    df_city = city_long.rename(columns={"geo_code": "city_geo_code"}).copy()
    df_city = df_city[["city_geo_code", "year", "indicator_code", "value", "flag"]]

    dfs = [df_city.assign(source="city")]
    if country_enriched_long is not None and not country_enriched_long.empty:
        df_c = country_enriched_long.copy()
        df_c["flag"] = None
        dfs.append(df_c[["city_geo_code", "year", "indicator_code", "value", "flag"]].assign(source="country"))

    long_all = pd.concat(dfs, ignore_index=True)
    long_all = long_all[(long_all["year"] >= year_min) & (long_all["year"] <= year_max)]

    expected = _expected_years_per_indicator(long_all, year_min, year_max)
    observed = (
        long_all.dropna(subset=["value"])
        .groupby(["city_geo_code", "indicator_code"])["year"]
        .nunique()
        .rename("n_years")
        .reset_index()
    )
    observed = observed.merge(expected.rename("expected_years"), on="indicator_code", how="left")
    observed["avail_ratio"] = observed["n_years"] / observed["expected_years"].replace(0, np.nan)

    # Aggregate coverage metrics
    agg = observed.groupby("city_geo_code").agg(
        n_indicators_with_any_data=("indicator_code", "nunique"),
        mean_avail_ratio=("avail_ratio", "mean"),
        median_avail_ratio=("avail_ratio", "median"),
        pct_indicators_ge_0_7=("avail_ratio", lambda s: float(np.mean(s >= 0.7)) if len(s) else 0.0),
        pct_indicators_ge_0_9=("avail_ratio", lambda s: float(np.mean(s >= 0.9)) if len(s) else 0.0),
    )

    # Flag rate (only city rows typically have flags)
    flag_df = long_all.copy()
    flag_df["has_flag"] = flag_df["flag"].notna() & (flag_df["flag"].astype(str).str.strip() != "")
    flag_rate = (
        flag_df.groupby("city_geo_code")["has_flag"].mean().rename("flag_rate").fillna(0.0)
    )

    out = agg.join(flag_rate, how="left").fillna(0.0)

    # Add top-N indicator missingness pattern features (availability ratios)
    # Pick indicators with highest overall variability in availability ratios.
    var_df = observed.groupby("indicator_code")["avail_ratio"].std().sort_values(ascending=False)
    top_inds = var_df.head(int(top_n)).index.tolist()
    if top_inds:
        wide = (
            observed[observed["indicator_code"].isin(top_inds)]
            .pivot_table(index="city_geo_code", columns="indicator_code", values="avail_ratio", aggfunc="mean")
        )
        wide.columns = [f"avail_{c}" for c in wide.columns]
        out = out.join(wide, how="left")

    out = out.reset_index().rename(columns={"city_geo_code": "geo_code"})
    return out


def _series_slope(years: np.ndarray, values: np.ndarray) -> float:
    # Simple linear regression slope using polyfit.
    if len(values) < 3:
        return float("nan")
    try:
        return float(np.polyfit(years.astype(float), values.astype(float), 1)[0])
    except Exception:
        return float("nan")


def build_typology_features(
    *,
    city_long: pd.DataFrame,
    country_enriched_long: Optional[pd.DataFrame],
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """
    Multi-year summary per city per indicator: mean, last, slope, volatility.
    Returns one row per city with wide features.
    """
    df_city = city_long.rename(columns={"geo_code": "city_geo_code"}).copy()
    df_city = df_city[["city_geo_code", "year", "indicator_code", "value"]]

    dfs = [df_city.assign(source="city")]
    if country_enriched_long is not None and not country_enriched_long.empty:
        dfs.append(country_enriched_long[["city_geo_code", "year", "indicator_code", "value"]].assign(source="country"))

    long_all = pd.concat(dfs, ignore_index=True)
    long_all = long_all[(long_all["year"] >= year_min) & (long_all["year"] <= year_max)]
    long_all["value"] = pd.to_numeric(long_all["value"], errors="coerce")

    def summarize(group: pd.DataFrame) -> pd.Series:
        g = group.dropna(subset=["value"]).sort_values("year")
        if g.empty:
            return pd.Series({"mean": np.nan, "last": np.nan, "slope": np.nan, "volatility": np.nan})
        years = g["year"].to_numpy()
        vals = g["value"].to_numpy()
        mean = float(np.nanmean(vals))
        last = float(vals[-1])
        slope = _series_slope(years, vals)
        # volatility: std of YoY % change (robust-ish)
        yoy = pd.Series(vals).pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        vol = float(np.nanstd(yoy)) if len(yoy) else float("nan")
        return pd.Series({"mean": mean, "last": last, "slope": slope, "volatility": vol})

    summary = (
        long_all.groupby(["city_geo_code", "indicator_code"], as_index=False)
        .apply(summarize)
        .reset_index(drop=True)
    )

    # Wide: feature names as {indicator}__{stat}
    wide = summary.pivot_table(
        index="city_geo_code",
        columns="indicator_code",
        values=["mean", "last", "slope", "volatility"],
        aggfunc="first",
    )
    wide.columns = [f"{ind}__{stat}" for stat, ind in wide.columns]
    wide = wide.reset_index().rename(columns={"city_geo_code": "geo_code"})
    return wide


def build_trajectory_features(
    *,
    city_long: pd.DataFrame,
    country_enriched_long: Optional[pd.DataFrame],
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """
    Trajectory representation: per-indicator slope, delta(last-first), volatility.
    Returns one row per city.
    """
    df_city = city_long.rename(columns={"geo_code": "city_geo_code"}).copy()
    df_city = df_city[["city_geo_code", "year", "indicator_code", "value"]]

    dfs = [df_city.assign(source="city")]
    if country_enriched_long is not None and not country_enriched_long.empty:
        dfs.append(country_enriched_long[["city_geo_code", "year", "indicator_code", "value"]].assign(source="country"))

    long_all = pd.concat(dfs, ignore_index=True)
    long_all = long_all[(long_all["year"] >= year_min) & (long_all["year"] <= year_max)]
    long_all["value"] = pd.to_numeric(long_all["value"], errors="coerce")

    def summarize(group: pd.DataFrame) -> pd.Series:
        g = group.dropna(subset=["value"]).sort_values("year")
        if len(g) < 2:
            return pd.Series({"slope": np.nan, "delta": np.nan, "volatility": np.nan})
        years = g["year"].to_numpy()
        vals = g["value"].to_numpy()
        slope = _series_slope(years, vals)
        delta = float(vals[-1] - vals[0])
        yoy = pd.Series(vals).pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        vol = float(np.nanstd(yoy)) if len(yoy) else float("nan")
        return pd.Series({"slope": slope, "delta": delta, "volatility": vol})

    summary = (
        long_all.groupby(["city_geo_code", "indicator_code"], as_index=False)
        .apply(summarize)
        .reset_index(drop=True)
    )

    wide = summary.pivot_table(
        index="city_geo_code",
        columns="indicator_code",
        values=["slope", "delta", "volatility"],
        aggfunc="first",
    )
    wide.columns = [f"{ind}__traj_{stat}" for stat, ind in wide.columns]
    wide = wide.reset_index().rename(columns={"city_geo_code": "geo_code"})
    return wide


