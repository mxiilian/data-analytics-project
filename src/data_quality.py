"""
Data Quality Assessment Script (Phase 5)
=========================================
Outlier detection, missing value analysis, and quality flagging.

Adapts detection methods and thresholds based on:
- Data source (Eurostat city, Eurostat country, WHO)
- Indicator type (rates, counts, concentrations)
- Expected variability patterns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Classification of indicator types for adapted thresholds."""
    RATE_PERCENT = "rate_percent"       # 0-100 range (recycling %, participation %)
    RATE_PER_CAPITA = "rate_per_capita" # Per 1000 or 100k (cars, hospital beds)
    COUNT_ABSOLUTE = "count_absolute"   # Raw counts (population, students)
    CONCENTRATION = "concentration"      # Air quality (PM2.5, NO2)
    COUNT_LARGE = "count_large"         # Large numbers (tourist nights)


@dataclass
class IndicatorConfig:
    """Configuration for outlier detection per indicator."""
    indicator: str
    indicator_type: IndicatorType
    value_column: str = "obs_value"
    # Thresholds (None = use defaults)
    zscore_threshold: Optional[float] = None
    iqr_multiplier: Optional[float] = None
    yoy_change_threshold: Optional[float] = None  # % change threshold
    min_valid: Optional[float] = None  # Minimum valid value
    max_valid: Optional[float] = None  # Maximum valid value


# Indicator-specific configurations
INDICATOR_CONFIGS: dict[str, IndicatorConfig] = {
    # City-level Eurostat
    "cars_per_1000": IndicatorConfig(
        indicator="cars_per_1000",
        indicator_type=IndicatorType.RATE_PER_CAPITA,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=15.0,
        min_valid=50,
        max_valid=800,
    ),
    "population_total": IndicatorConfig(
        indicator="population_total",
        indicator_type=IndicatorType.COUNT_ABSOLUTE,
        zscore_threshold=3.0,  # Cities vary widely
        iqr_multiplier=2.0,
        yoy_change_threshold=5.0,  # Population shouldn't change fast
        min_valid=100_000,
    ),
    "tourist_nights": IndicatorConfig(
        indicator="tourist_nights",
        indicator_type=IndicatorType.COUNT_LARGE,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=50.0,  # COVID caused huge swings
        min_valid=0,
    ),
    "higher_ed_students": IndicatorConfig(
        indicator="higher_ed_students",
        indicator_type=IndicatorType.COUNT_ABSOLUTE,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=15.0,
        min_valid=0,
    ),
    "unemployed_total": IndicatorConfig(
        indicator="unemployed_total",
        indicator_type=IndicatorType.COUNT_ABSOLUTE,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=30.0,  # Can spike in crises
        min_valid=0,
    ),
    "employed_20_64": IndicatorConfig(
        indicator="employed_20_64",
        indicator_type=IndicatorType.COUNT_ABSOLUTE,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=10.0,
        min_valid=0,
    ),
    "foreign_population": IndicatorConfig(
        indicator="foreign_population",
        indicator_type=IndicatorType.COUNT_ABSOLUTE,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=15.0,
        min_valid=0,
    ),
    # Country-level Eurostat
    "hospital_beds_per_100k": IndicatorConfig(
        indicator="hospital_beds_per_100k",
        indicator_type=IndicatorType.RATE_PER_CAPITA,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=10.0,
        min_valid=100,
        max_valid=1500,
    ),
    "recycling_rate_pct": IndicatorConfig(
        indicator="recycling_rate_pct",
        indicator_type=IndicatorType.RATE_PERCENT,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=10.0,
        min_valid=0,
        max_valid=100,
    ),
    "internet_vhcn_pct": IndicatorConfig(
        indicator="internet_vhcn_pct",
        indicator_type=IndicatorType.RATE_PERCENT,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=20.0,  # Can grow fast
        min_valid=0,
        max_valid=100,
    ),
    "education_participation_pct": IndicatorConfig(
        indicator="education_participation_pct",
        indicator_type=IndicatorType.RATE_PERCENT,
        zscore_threshold=2.5,
        iqr_multiplier=1.5,
        yoy_change_threshold=15.0,
        min_valid=0,
        max_valid=100,
    ),
    # WHO Air Quality - separate columns
    "pm10_concentration": IndicatorConfig(
        indicator="pm10_concentration",
        indicator_type=IndicatorType.CONCENTRATION,
        value_column="pm10_concentration",
        zscore_threshold=2.0,  # Stricter for health data
        iqr_multiplier=1.5,
        yoy_change_threshold=30.0,
        min_valid=0,
        max_valid=200,  # Extremely high would be suspicious
    ),
    "pm25_concentration": IndicatorConfig(
        indicator="pm25_concentration",
        indicator_type=IndicatorType.CONCENTRATION,
        value_column="pm25_concentration",
        zscore_threshold=2.0,
        iqr_multiplier=1.5,
        yoy_change_threshold=30.0,
        min_valid=0,
        max_valid=100,
    ),
    "no2_concentration": IndicatorConfig(
        indicator="no2_concentration",
        indicator_type=IndicatorType.CONCENTRATION,
        value_column="no2_concentration",
        zscore_threshold=2.0,
        iqr_multiplier=1.5,
        yoy_change_threshold=30.0,
        min_valid=0,
        max_valid=150,
    ),
}

# Default thresholds by indicator type
DEFAULT_THRESHOLDS: dict[IndicatorType, dict] = {
    IndicatorType.RATE_PERCENT: {
        "zscore": 2.5,
        "iqr": 1.5,
        "yoy": 15.0,
    },
    IndicatorType.RATE_PER_CAPITA: {
        "zscore": 2.5,
        "iqr": 1.5,
        "yoy": 15.0,
    },
    IndicatorType.COUNT_ABSOLUTE: {
        "zscore": 3.0,
        "iqr": 2.0,
        "yoy": 20.0,
    },
    IndicatorType.CONCENTRATION: {
        "zscore": 2.0,
        "iqr": 1.5,
        "yoy": 30.0,
    },
    IndicatorType.COUNT_LARGE: {
        "zscore": 2.5,
        "iqr": 1.5,
        "yoy": 50.0,
    },
}


# -----------------------------------------------------------------------------
# Outlier Detection Methods
# -----------------------------------------------------------------------------

def compute_zscore(series: pd.Series) -> pd.Series:
    """Compute standard Z-score: (x - mean) / std."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def compute_modified_zscore(series: pd.Series) -> pd.Series:
    """
    Compute Modified Z-score using median and MAD.
    More robust to outliers than standard Z-score.
    
    Formula: 0.6745 * (x - median) / MAD
    """
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (series - median) / mad


def compute_iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Compute IQR-based outlier bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def compute_yoy_change(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    year_col: str = "year",
) -> pd.Series:
    """
    Compute year-over-year percentage change.
    
    Returns percentage change from previous year within each group.
    """
    df = df.sort_values([group_col, year_col])
    
    # Compute pct_change per group without deprecated apply
    return df.groupby(group_col)[value_col].pct_change() * 100


# -----------------------------------------------------------------------------
# Quality Assessment Functions
# -----------------------------------------------------------------------------

@dataclass
class OutlierResult:
    """Result of outlier detection for a single value."""
    is_outlier_zscore: bool = False
    is_outlier_modified_zscore: bool = False
    is_outlier_iqr: bool = False
    is_outlier_range: bool = False
    is_outlier_temporal: bool = False
    zscore: float = 0.0
    modified_zscore: float = 0.0
    yoy_change_pct: Optional[float] = None
    
    @property
    def is_any_outlier(self) -> bool:
        return any([
            self.is_outlier_zscore,
            self.is_outlier_modified_zscore,
            self.is_outlier_iqr,
            self.is_outlier_range,
            self.is_outlier_temporal,
        ])
    
    @property
    def outlier_methods(self) -> list[str]:
        methods = []
        if self.is_outlier_zscore:
            methods.append("zscore")
        if self.is_outlier_modified_zscore:
            methods.append("mod_zscore")
        if self.is_outlier_iqr:
            methods.append("iqr")
        if self.is_outlier_range:
            methods.append("range")
        if self.is_outlier_temporal:
            methods.append("temporal")
        return methods


def assess_indicator_quality(
    df: pd.DataFrame,
    config: IndicatorConfig,
    location_col: str = "city_name",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Assess data quality for a single indicator.
    
    Returns DataFrame with outlier flags and quality metrics.
    """
    df = df.copy()
    value_col = config.value_column
    
    if value_col not in df.columns:
        logger.warning(f"Value column '{value_col}' not found")
        return df
    
    # Get thresholds
    defaults = DEFAULT_THRESHOLDS.get(
        config.indicator_type,
        DEFAULT_THRESHOLDS[IndicatorType.COUNT_ABSOLUTE]
    )
    zscore_thresh = config.zscore_threshold or defaults["zscore"]
    iqr_mult = config.iqr_multiplier or defaults["iqr"]
    yoy_thresh = config.yoy_change_threshold or defaults["yoy"]
    
    # Extract numeric values
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid_mask = values.notna()
    
    # Initialize result columns
    df["is_missing"] = ~valid_mask
    df["zscore"] = np.nan
    df["modified_zscore"] = np.nan
    df["is_outlier_zscore"] = False
    df["is_outlier_mod_zscore"] = False
    df["is_outlier_iqr"] = False
    df["is_outlier_range"] = False
    df["is_outlier_temporal"] = False
    df["yoy_change_pct"] = np.nan
    
    if valid_mask.sum() < 3:
        logger.warning(f"Too few valid values for {config.indicator}")
        return df
    
    valid_values = values[valid_mask]
    
    # Z-score detection
    zscores = compute_zscore(valid_values)
    df.loc[valid_mask, "zscore"] = zscores
    df.loc[valid_mask, "is_outlier_zscore"] = np.abs(zscores) > zscore_thresh
    
    # Modified Z-score detection
    mod_zscores = compute_modified_zscore(valid_values)
    df.loc[valid_mask, "modified_zscore"] = mod_zscores
    df.loc[valid_mask, "is_outlier_mod_zscore"] = np.abs(mod_zscores) > zscore_thresh
    
    # IQR detection
    lower, upper = compute_iqr_bounds(valid_values, iqr_mult)
    df.loc[valid_mask, "is_outlier_iqr"] = (valid_values < lower) | (valid_values > upper)
    df["iqr_lower"] = lower
    df["iqr_upper"] = upper
    
    # Range validation
    if config.min_valid is not None:
        df.loc[valid_mask & (values < config.min_valid), "is_outlier_range"] = True
    if config.max_valid is not None:
        df.loc[valid_mask & (values > config.max_valid), "is_outlier_range"] = True
    
    # Temporal outlier detection (year-over-year)
    if location_col in df.columns and year_col in df.columns:
        yoy = compute_yoy_change(df, value_col, location_col, year_col)
        df["yoy_change_pct"] = yoy
        df["is_outlier_temporal"] = np.abs(yoy) > yoy_thresh
    
    # Aggregate outlier flag
    df["is_any_outlier"] = (
        df["is_outlier_zscore"] |
        df["is_outlier_mod_zscore"] |
        df["is_outlier_iqr"] |
        df["is_outlier_range"] |
        df["is_outlier_temporal"]
    )
    
    # Build outlier method string
    def get_outlier_methods(row):
        methods = []
        if row["is_outlier_zscore"]:
            methods.append("zscore")
        if row["is_outlier_mod_zscore"]:
            methods.append("mod_zscore")
        if row["is_outlier_iqr"]:
            methods.append("iqr")
        if row["is_outlier_range"]:
            methods.append("range")
        if row["is_outlier_temporal"]:
            methods.append("temporal")
        return ",".join(methods) if methods else ""
    
    df["outlier_methods"] = df.apply(get_outlier_methods, axis=1)
    
    return df


# -----------------------------------------------------------------------------
# Summary Generation
# -----------------------------------------------------------------------------

@dataclass
class IndicatorQualitySummary:
    """Summary statistics for an indicator's quality assessment."""
    indicator: str
    total_rows: int
    missing_count: int
    missing_pct: float
    outlier_count: int
    outlier_pct: float
    outlier_zscore_count: int
    outlier_iqr_count: int
    outlier_temporal_count: int
    outlier_range_count: int
    mean_value: Optional[float]
    std_value: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    cities_with_outliers: list[str] = field(default_factory=list)


def generate_indicator_summary(
    df: pd.DataFrame,
    indicator: str,
    value_col: str = "obs_value",
    location_col: str = "city_name",
) -> IndicatorQualitySummary:
    """Generate quality summary for a single indicator."""
    total = len(df)
    
    # Use appropriate location column
    if location_col not in df.columns:
        location_col = "country_code" if "country_code" in df.columns else "location_code"
    
    missing = df["is_missing"].sum() if "is_missing" in df.columns else df[value_col].isna().sum()
    
    outlier_count = df["is_any_outlier"].sum() if "is_any_outlier" in df.columns else 0
    
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid_values = values.dropna()
    
    # Get cities with outliers
    cities_with_outliers = []
    if "is_any_outlier" in df.columns and location_col in df.columns:
        cities_with_outliers = df.loc[df["is_any_outlier"], location_col].unique().tolist()
    
    return IndicatorQualitySummary(
        indicator=indicator,
        total_rows=total,
        missing_count=int(missing),
        missing_pct=round(100 * missing / total, 1) if total > 0 else 0,
        outlier_count=int(outlier_count),
        outlier_pct=round(100 * outlier_count / total, 1) if total > 0 else 0,
        outlier_zscore_count=int(df["is_outlier_zscore"].sum()) if "is_outlier_zscore" in df.columns else 0,
        outlier_iqr_count=int(df["is_outlier_iqr"].sum()) if "is_outlier_iqr" in df.columns else 0,
        outlier_temporal_count=int(df["is_outlier_temporal"].sum()) if "is_outlier_temporal" in df.columns else 0,
        outlier_range_count=int(df["is_outlier_range"].sum()) if "is_outlier_range" in df.columns else 0,
        mean_value=round(valid_values.mean(), 2) if len(valid_values) > 0 else None,
        std_value=round(valid_values.std(), 2) if len(valid_values) > 0 else None,
        min_value=round(valid_values.min(), 2) if len(valid_values) > 0 else None,
        max_value=round(valid_values.max(), 2) if len(valid_values) > 0 else None,
        cities_with_outliers=cities_with_outliers,
    )


def generate_compact_summary(summaries: list[IndicatorQualitySummary]) -> str:
    """Generate a compact text summary of all quality assessments."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATA QUALITY ASSESSMENT SUMMARY")
    lines.append("=" * 70)
    
    # Overall stats
    total_rows = sum(s.total_rows for s in summaries)
    total_missing = sum(s.missing_count for s in summaries)
    total_outliers = sum(s.outlier_count for s in summaries)
    
    lines.append("\nOVERALL STATISTICS")
    lines.append(f"  Total data points:     {total_rows:,}")
    lines.append(f"  Total missing values:  {total_missing:,} ({100*total_missing/total_rows:.1f}%)")
    lines.append(f"  Total outliers:        {total_outliers:,} ({100*total_outliers/total_rows:.1f}%)")
    
    # Per-indicator table
    lines.append("\nPER-INDICATOR SUMMARY")
    lines.append("-" * 70)
    header = f"{'Indicator':<28} {'Rows':>6} {'Miss%':>6} {'Out%':>6} {'Z':>4} {'IQR':>4} {'YoY':>4}"
    lines.append(header)
    lines.append("-" * 70)
    
    for s in sorted(summaries, key=lambda x: x.outlier_pct, reverse=True):
        row = (
            f"{s.indicator[:28]:<28} "
            f"{s.total_rows:>6} "
            f"{s.missing_pct:>5.1f}% "
            f"{s.outlier_pct:>5.1f}% "
            f"{s.outlier_zscore_count:>4} "
            f"{s.outlier_iqr_count:>4} "
            f"{s.outlier_temporal_count:>4}"
        )
        lines.append(row)
    
    lines.append("-" * 70)
    
    # Top outlier locations
    lines.append("\nLOCATIONS WITH MOST OUTLIERS")
    all_outlier_locations = []
    for s in summaries:
        all_outlier_locations.extend(s.cities_with_outliers)
    
    from collections import Counter
    location_counts = Counter(all_outlier_locations)
    for location, count in location_counts.most_common(10):
        lines.append(f"  {location}: {count} outliers across indicators")
    
    # Indicators needing attention
    high_missing = [s for s in summaries if s.missing_pct > 10]
    high_outlier = [s for s in summaries if s.outlier_pct > 15]
    
    if high_missing:
        lines.append("\n⚠️  HIGH MISSING DATA (>10%):")
        for s in high_missing:
            lines.append(f"    - {s.indicator}: {s.missing_pct:.1f}% missing")
    
    if high_outlier:
        lines.append("\n⚠️  HIGH OUTLIER RATE (>15%):")
        for s in high_outlier:
            lines.append(f"    - {s.indicator}: {s.outlier_pct:.1f}% outliers")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Processing
# -----------------------------------------------------------------------------

def process_eurostat_file(
    file_path: Path,
    indicator_name: str,
) -> tuple[pd.DataFrame, IndicatorQualitySummary]:
    """Process a single Eurostat processed file."""
    df = pd.read_csv(file_path)
    
    # Get config or create default
    config = INDICATOR_CONFIGS.get(
        indicator_name,
        IndicatorConfig(
            indicator=indicator_name,
            indicator_type=IndicatorType.COUNT_ABSOLUTE,
        )
    )
    
    # Determine location column
    location_col = "city_name" if "city_name" in df.columns else "country_code"
    
    # Assess quality
    df_assessed = assess_indicator_quality(df, config, location_col=location_col)
    
    # Generate summary
    summary = generate_indicator_summary(
        df_assessed,
        indicator_name,
        value_col=config.value_column,
        location_col=location_col,
    )
    
    return df_assessed, summary


def process_who_file(file_path: Path) -> tuple[pd.DataFrame, list[IndicatorQualitySummary]]:
    """Process WHO air quality file (multiple pollutant columns)."""
    df = pd.read_csv(file_path)
    summaries = []
    
    pollutants = ["pm10_concentration", "pm25_concentration", "no2_concentration"]
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            continue
        
        config = INDICATOR_CONFIGS.get(
            pollutant,
            IndicatorConfig(
                indicator=pollutant,
                indicator_type=IndicatorType.CONCENTRATION,
                value_column=pollutant,
            )
        )
        
        # Assess this pollutant
        df = assess_indicator_quality(
            df, config,
            location_col="city_name",
            year_col="year",
        )
        
        # Rename outlier columns to be pollutant-specific
        outlier_cols = [
            "is_outlier_zscore", "is_outlier_mod_zscore", "is_outlier_iqr",
            "is_outlier_range", "is_outlier_temporal", "is_any_outlier",
            "zscore", "modified_zscore", "yoy_change_pct", "outlier_methods",
        ]
        for col in outlier_cols:
            if col in df.columns:
                df = df.rename(columns={col: f"{pollutant}_{col}"})
        
        # Generate summary
        summary = generate_indicator_summary(
            df, pollutant,
            value_col=pollutant,
            location_col="city_name",
        )
        # Fix: re-read outlier counts from renamed columns
        summary.outlier_count = int(df[f"{pollutant}_is_any_outlier"].sum()) if f"{pollutant}_is_any_outlier" in df.columns else 0
        summary.outlier_pct = round(100 * summary.outlier_count / summary.total_rows, 1) if summary.total_rows > 0 else 0
        
        summaries.append(summary)
    
    return df, summaries


def run_quality_assessment(
    processed_dir: Path,
    output_dir: Path,
) -> None:
    """Run quality assessment on all processed files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_summaries: list[IndicatorQualitySummary] = []
    all_outliers: list[pd.DataFrame] = []
    
    # Mapping of files to indicator names
    file_indicator_map = {
        "urb_ctran_processed.csv": "cars_per_1000",
        "urb_cpop1_processed.csv": "population_total",
        "urb_ctour_processed.csv": "tourist_nights",
        "urb_ceduc_processed.csv": "higher_ed_students",
        "urb_clma_unemployed_processed.csv": "unemployed_total",
        "urb_clma_employed_processed.csv": "employed_20_64",
        "urb_cpopcb_processed.csv": "foreign_population",
        "hlth_rs_bds1_processed.csv": "hospital_beds_per_100k",
        "sdg_11_60_processed.csv": "recycling_rate_pct",
        "sdg_17_60_processed.csv": "internet_vhcn_pct",
        "trng_lfse_01_processed.csv": "education_participation_pct",
    }
    
    # Process Eurostat files
    for filename, indicator_name in file_indicator_map.items():
        file_path = processed_dir / filename
        if not file_path.exists():
            logger.warning(f"File not found: {filename}")
            continue
        
        logger.info(f"Processing: {filename}")
        df_assessed, summary = process_eurostat_file(file_path, indicator_name)
        all_summaries.append(summary)
        
        # Save assessed file
        output_path = output_dir / f"{indicator_name}_quality.csv"
        df_assessed.to_csv(output_path, index=False)
        
        # Collect outliers
        if "is_any_outlier" in df_assessed.columns:
            outliers = df_assessed[df_assessed["is_any_outlier"]].copy()
            outliers["source_indicator"] = indicator_name
            all_outliers.append(outliers)
    
    # Process WHO file
    who_path = processed_dir / "who_air_quality_processed.csv"
    if who_path.exists():
        logger.info("Processing: who_air_quality_processed.csv")
        df_who, who_summaries = process_who_file(who_path)
        all_summaries.extend(who_summaries)
        
        # Save assessed WHO file
        df_who.to_csv(output_dir / "who_air_quality_quality.csv", index=False)
    
    # Generate and save compact summary
    summary_text = generate_compact_summary(all_summaries)
    print(summary_text)
    
    with open(output_dir / "quality_summary.txt", "w") as f:
        f.write(summary_text)
    
    # Save summary as CSV
    summary_records = []
    for s in all_summaries:
        summary_records.append({
            "indicator": s.indicator,
            "total_rows": s.total_rows,
            "missing_count": s.missing_count,
            "missing_pct": s.missing_pct,
            "outlier_count": s.outlier_count,
            "outlier_pct": s.outlier_pct,
            "outlier_zscore": s.outlier_zscore_count,
            "outlier_iqr": s.outlier_iqr_count,
            "outlier_temporal": s.outlier_temporal_count,
            "outlier_range": s.outlier_range_count,
            "mean": s.mean_value,
            "std": s.std_value,
            "min": s.min_value,
            "max": s.max_value,
            "locations_with_outliers": ";".join(s.cities_with_outliers),
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(output_dir / "quality_summary.csv", index=False)
    
    # Save all outliers to single file
    if all_outliers:
        # Select common columns for outlier report
        outlier_cols = [
            "source_indicator", "city_name", "country_code", "year",
            "obs_value", "zscore", "modified_zscore", "yoy_change_pct",
            "outlier_methods",
        ]
        
        combined_outliers = []
        for df in all_outliers:
            available_cols = [c for c in outlier_cols if c in df.columns]
            combined_outliers.append(df[available_cols])
        
        if combined_outliers:
            outliers_df = pd.concat(combined_outliers, ignore_index=True)
            outliers_df.to_csv(output_dir / "all_outliers.csv", index=False)
            logger.info(f"Found {len(outliers_df)} total outlier records")
    
    logger.info(f"Quality assessment complete. Results saved to {output_dir}")


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Quality Assessment (Phase 5)"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output" / "processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output" / "quality",
        help="Path to quality output directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_quality_assessment(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )

