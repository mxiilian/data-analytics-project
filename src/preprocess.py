"""
Preprocessing Pipeline for European City Data Analysis
=======================================================
Phases 1-3: Column Cleanup, Geographic Filtering, Temporal Standardization

Target Cities:
    Copenhagen, Hamburg, Paris, Amsterdam, Stockholm, Budapest, Lisbon, Milan, Barcelona

Target Countries:
    Denmark, Germany, France, Netherlands, Sweden, Hungary, Portugal, Italy, Spain
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

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


class DatasetType(Enum):
    """Classification of dataset sources."""
    EUROSTAT_CITY = "eurostat_city"      # urb_* files (city-level)
    EUROSTAT_COUNTRY = "eurostat_country" # Other Eurostat files (country-level)
    WHO_AIR_QUALITY = "who_air_quality"   # WHO air quality database


@dataclass(frozen=True)
class CityMapping:
    """Mapping between city identifiers across datasets."""
    name: str                    # Display name
    eurostat_code: Optional[str] # Eurostat city code (e.g., DE002C)
    who_names: tuple[str, ...]   # WHO city name variants
    country_code: str            # ISO 2-letter country code
    country_name: str            # Full country name

    def matches_eurostat(self, code: str) -> bool:
        """Check if a code matches this city's Eurostat identifier."""
        return self.eurostat_code is not None and code == self.eurostat_code

    def matches_who(self, city_name: str) -> bool:
        """Check if a WHO city name matches this city."""
        # WHO format is typically "CityName/ISO3"
        base_name = city_name.split("/")[0].lower()
        return any(variant.lower() == base_name for variant in self.who_names)


# Target cities with their identifiers across datasets
# Note: Copenhagen (DK) is NOT in Eurostat city datasets
TARGET_CITIES: tuple[CityMapping, ...] = (
    CityMapping(
        name="Copenhagen",
        eurostat_code=None,  # Not available in Eurostat city data
        who_names=("Copenhagen", "Kobenhavn", "København"),
        country_code="DK",
        country_name="Denmark",
    ),
    CityMapping(
        name="Hamburg",
        eurostat_code="DE002C",
        who_names=("Hamburg",),
        country_code="DE",
        country_name="Germany",
    ),
    CityMapping(
        name="Paris",
        eurostat_code="FR001C",
        who_names=("Paris",),
        country_code="FR",
        country_name="France",
    ),
    CityMapping(
        name="Amsterdam",
        eurostat_code="NL001C",
        who_names=("Amsterdam",),
        country_code="NL",
        country_name="Netherlands",
    ),
    CityMapping(
        name="Stockholm",
        eurostat_code="SE001C",
        who_names=("Stockholm",),
        country_code="SE",
        country_name="Sweden",
    ),
    CityMapping(
        name="Budapest",
        eurostat_code="HU001C",
        who_names=("Budapest",),
        country_code="HU",
        country_name="Hungary",
    ),
    CityMapping(
        name="Lisbon",
        eurostat_code="PT001C",
        who_names=("Lisboa", "Lisbon"),
        country_code="PT",
        country_name="Portugal",
    ),
    CityMapping(
        name="Milan",
        eurostat_code="IT002C",
        who_names=("Milano", "Milan"),
        country_code="IT",
        country_name="Italy",
    ),
    CityMapping(
        name="Barcelona",
        eurostat_code="ES002C",
        who_names=("Barcelona",),
        country_code="ES",
        country_name="Spain",
    ),
)

# Country codes for filtering country-level datasets
TARGET_COUNTRY_CODES: frozenset[str] = frozenset(
    city.country_code for city in TARGET_CITIES
)

# Eurostat city codes for filtering city-level datasets
TARGET_EUROSTAT_CITY_CODES: frozenset[str] = frozenset(
    city.eurostat_code for city in TARGET_CITIES if city.eurostat_code
)

# Analysis time period
YEAR_MIN = 2015
YEAR_MAX = 2023


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    filename: str
    dataset_type: DatasetType
    indicator_name: str
    indicator_description: str
    location_column: str = "geo"  # Column containing location codes
    value_column: str = "OBS_VALUE"
    time_column: str = "TIME_PERIOD"
    extra_filters: dict = field(default_factory=dict)


# Dataset configurations based on notebook analysis
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # City-level datasets (urb_*)
    "urb_ctran": DatasetConfig(
        filename="urb_ctran_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="cars_per_1000",
        indicator_description="Number of registered cars per 1000 population",
        location_column="cities",
    ),
    "urb_cpop1": DatasetConfig(
        filename="urb_cpop1_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="population_total",
        indicator_description="Population on 1 January, total",
        location_column="cities",
    ),
    "urb_ctour": DatasetConfig(
        filename="urb_ctour_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="tourist_nights",
        indicator_description="Total nights spent in tourist accommodation",
        location_column="cities",
    ),
    "urb_ceduc": DatasetConfig(
        filename="urb_ceduc_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="higher_ed_students",
        indicator_description="Students in higher education (ISCED 5-8)",
        location_column="cities",
    ),
    "urb_clma_unemployed": DatasetConfig(
        filename="urb_clma_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="unemployed_total",
        indicator_description="Persons unemployed, total",
        location_column="cities",
    ),
    "urb_clma_employed": DatasetConfig(
        filename="urb_clma_page_linear_2_0(1).csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="employed_20_64",
        indicator_description="Persons employed, 20-64, total",
        location_column="cities",
    ),
    "urb_cpopcb": DatasetConfig(
        filename="urb_cpopcb_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_CITY,
        indicator_name="foreign_population",
        indicator_description="Population by citizenship - foreigners",
        location_column="cities",
    ),
    # Country-level datasets
    "hlth_rs_bds1": DatasetConfig(
        filename="hlth_rs_bds1_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_COUNTRY,
        indicator_name="hospital_beds_per_100k",
        indicator_description="Hospital beds per 100,000 inhabitants",
        location_column="geo",
    ),
    "sdg_11_60": DatasetConfig(
        filename="sdg_11_60_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_COUNTRY,
        indicator_name="recycling_rate_pct",
        indicator_description="Recycling rate of municipal waste (%)",
        location_column="geo",
    ),
    "sdg_17_60": DatasetConfig(
        filename="sdg_17_60_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_COUNTRY,
        indicator_name="internet_vhcn_pct",
        indicator_description="High-speed internet coverage (VHCN) % households",
        location_column="geo",
    ),
    "trng_lfse_01": DatasetConfig(
        filename="trng_lfse_01_page_linear_2_0.csv",
        dataset_type=DatasetType.EUROSTAT_COUNTRY,
        indicator_name="education_participation_pct",
        indicator_description="Participation rate in education/training, 25-74 years",
        location_column="geo",
    ),
    # WHO Air Quality (special structure)
    "who_air_quality": DatasetConfig(
        filename="who_ambient_air_quality_database_version_2024_(v6.1).csv",
        dataset_type=DatasetType.WHO_AIR_QUALITY,
        indicator_name="air_quality",
        indicator_description="Air quality concentrations (PM10, PM2.5, NO2)",
        location_column="city",
        value_column="pm25_concentration",  # Primary, others handled separately
        time_column="year",
    ),
}

# Columns to drop from Eurostat datasets (redundant/metadata)
EUROSTAT_COLUMNS_TO_DROP: tuple[str, ...] = (
    # Redundant value columns
    "Time",
    "Observation value",
    # Redundant flag columns
    "Observation status (Flag) V2 structure",
    "Confidentiality status (flag)",
    # Metadata columns
    "STRUCTURE",
    "STRUCTURE_ID",
    "STRUCTURE_NAME",
    # Verbose label columns (keep codes, drop labels)
    "Time frequency",
    "Urban audit indicator",
    "Geopolitical entity (declaring)",
    "Geopolitical entity (reporting)",
    "Unit of measure",
)


# -----------------------------------------------------------------------------
# Phase 1: Column Cleanup
# -----------------------------------------------------------------------------

def cleanup_eurostat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Remove redundant columns from Eurostat data.
    
    Keeps essential columns:
    - OBS_VALUE (observation value)
    - TIME_PERIOD (time)
    - geo/cities (location identifier)
    - OBS_FLAG, CONF_STATUS (quality flags - short form)
    - Indicator-specific columns (freq, indic_ur, etc.)
    """
    # Identify columns to drop that exist in this dataframe
    cols_to_drop = [col for col in EUROSTAT_COLUMNS_TO_DROP if col in df.columns]
    
    if cols_to_drop:
        logger.debug(f"Dropping {len(cols_to_drop)} redundant columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def standardize_column_names(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Rename columns to a consistent schema.
    
    Standard schema:
    - obs_value: The observation value
    - time_period: Year or time period
    - location_code: City code or country code
    - obs_flag: Observation quality flag
    - conf_status: Confidentiality status
    """
    rename_map = {
        config.value_column: "obs_value",
        config.time_column: "time_period",
        config.location_column: "location_code",
    }
    
    # Optional columns
    if "OBS_FLAG" in df.columns:
        rename_map["OBS_FLAG"] = "obs_flag"
    if "CONF_STATUS" in df.columns:
        rename_map["CONF_STATUS"] = "conf_status"
    
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    
    return df.rename(columns=rename_map)


# -----------------------------------------------------------------------------
# Phase 2: Geographic Filtering
# -----------------------------------------------------------------------------

def is_city_code(code: str) -> bool:
    """
    Check if a location code is a city (not a country aggregate).
    
    Eurostat city codes typically end with 'C' or 'K':
    - xxxC = City
    - xxxK = Greater city / Capital
    
    Country codes are 2-letter ISO codes (DE, FR, etc.)
    """
    if pd.isna(code):
        return False
    code = str(code).strip()
    # City codes are longer and end with C or K
    return len(code) > 2 and code[-1] in ("C", "K")


def filter_eurostat_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Eurostat city data to only target cities.
    
    Removes:
    - Country aggregates (2-letter codes like 'DE', 'FR')
    - Cities not in target list
    """
    if "location_code" not in df.columns:
        logger.warning("No location_code column found for city filtering")
        return df
    
    # First, filter to only city codes (remove country aggregates)
    city_mask = df["location_code"].apply(is_city_code)
    df_cities = df[city_mask].copy()
    
    removed_aggregates = len(df) - len(df_cities)
    if removed_aggregates > 0:
        logger.info(f"Removed {removed_aggregates} country aggregate rows")
    
    # Then, filter to target cities
    target_mask = df_cities["location_code"].isin(TARGET_EUROSTAT_CITY_CODES)
    df_filtered = df_cities[target_mask].copy()
    
    logger.info(
        f"Filtered to {len(df_filtered)} rows for target cities "
        f"(from {len(df_cities)} city rows)"
    )
    
    return df_filtered


def filter_eurostat_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Eurostat country data to only target countries.
    """
    if "location_code" not in df.columns:
        logger.warning("No location_code column found for country filtering")
        return df
    
    target_mask = df["location_code"].isin(TARGET_COUNTRY_CODES)
    df_filtered = df[target_mask].copy()
    
    logger.info(
        f"Filtered to {len(df_filtered)} rows for target countries "
        f"(from {len(df)} total rows)"
    )
    
    return df_filtered


def filter_who_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter WHO air quality data to target cities.
    
    WHO uses city names like 'Hamburg/DEU', 'Paris/FRA'.
    """
    if "location_code" not in df.columns:
        logger.warning("No location_code column found for WHO filtering")
        return df
    
    def matches_target_city(city_name: str) -> bool:
        if pd.isna(city_name):
            return False
        return any(city.matches_who(city_name) for city in TARGET_CITIES)
    
    target_mask = df["location_code"].apply(matches_target_city)
    df_filtered = df[target_mask].copy()
    
    logger.info(
        f"Filtered WHO data to {len(df_filtered)} rows for target cities "
        f"(from {len(df)} total rows)"
    )
    
    return df_filtered


def add_city_name_column(df: pd.DataFrame, dataset_type: DatasetType) -> pd.DataFrame:
    """
    Add a standardized city_name column based on location codes.
    """
    if "location_code" not in df.columns:
        return df
    
    def get_city_name(code: str) -> Optional[str]:
        if pd.isna(code):
            return None
        code = str(code).strip()
        
        if dataset_type == DatasetType.WHO_AIR_QUALITY:
            # WHO format: "CityName/ISO3"
            for city in TARGET_CITIES:
                if city.matches_who(code):
                    return city.name
        else:
            # Eurostat format: city codes like DE002C
            for city in TARGET_CITIES:
                if city.matches_eurostat(code):
                    return city.name
        return None
    
    df = df.copy()
    df["city_name"] = df["location_code"].apply(get_city_name)
    
    return df


def add_country_columns(df: pd.DataFrame, dataset_type: DatasetType) -> pd.DataFrame:
    """
    Add country_code and country_name columns based on location.
    """
    if "location_code" not in df.columns:
        return df
    
    df = df.copy()
    
    if dataset_type == DatasetType.EUROSTAT_COUNTRY:
        # Location code IS the country code
        df["country_code"] = df["location_code"]
        country_map = {city.country_code: city.country_name for city in TARGET_CITIES}
        df["country_name"] = df["country_code"].map(country_map)
    else:
        # Extract country from city code or WHO format
        def extract_country(code: str) -> tuple[Optional[str], Optional[str]]:
            if pd.isna(code):
                return None, None
            code = str(code).strip()
            
            if dataset_type == DatasetType.WHO_AIR_QUALITY:
                for city in TARGET_CITIES:
                    if city.matches_who(code):
                        return city.country_code, city.country_name
            else:
                for city in TARGET_CITIES:
                    if city.matches_eurostat(code):
                        return city.country_code, city.country_name
            return None, None
        
        country_info = df["location_code"].apply(extract_country)
        df["country_code"] = country_info.apply(lambda x: x[0])
        df["country_name"] = country_info.apply(lambda x: x[1])
    
    return df


# -----------------------------------------------------------------------------
# Phase 3: Temporal Standardization
# -----------------------------------------------------------------------------

def extract_year(time_value: str | int | float) -> Optional[int]:
    """
    Extract year from various time period formats.
    
    Handles:
    - Integer years: 2020
    - String years: "2020"
    - Monthly format: "2024-M11" or "2024-11"
    - Date format: "2020-01-01"
    """
    if pd.isna(time_value):
        return None
    
    time_str = str(time_value).strip()
    
    # Try direct integer conversion first
    try:
        year = int(float(time_str))
        if 1900 <= year <= 2100:
            return year
    except ValueError:
        pass
    
    # Try extracting year from various formats
    patterns = [
        r"^(\d{4})",           # Starts with 4 digits (2020, 2020-M11, etc.)
        r"(\d{4})-\d{2}-\d{2}",  # Date format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, time_str)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
    
    logger.warning(f"Could not extract year from: {time_str}")
    return None


def standardize_time_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 3: Standardize time periods to years.
    
    - Extracts year from various formats
    - Filters to target year range (YEAR_MIN to YEAR_MAX)
    - Adds 'year' column
    """
    if "time_period" not in df.columns:
        logger.warning("No time_period column found")
        return df
    
    df = df.copy()
    
    # Extract year
    df["year"] = df["time_period"].apply(extract_year)
    
    # Log extraction results
    null_years = df["year"].isna().sum()
    if null_years > 0:
        logger.warning(f"{null_years} rows have unparseable time periods")
    
    # Filter to target year range
    original_len = len(df)
    df = df[df["year"].notna()].copy()
    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    
    filtered_count = original_len - len(df)
    if filtered_count > 0:
        logger.info(
            f"Filtered {filtered_count} rows outside year range "
            f"[{YEAR_MIN}, {YEAR_MAX}]"
        )
    
    # Convert year to integer
    df["year"] = df["year"].astype(int)
    
    return df


def aggregate_to_yearly(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate sub-yearly data (monthly, quarterly) to yearly.
    
    For numeric values, takes the mean.
    For the first occurrence of other columns.
    """
    if "year" not in df.columns:
        logger.warning("No year column found for aggregation")
        return df
    
    # Identify columns to aggregate
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    value_cols = [c for c in numeric_cols if c not in group_cols and c != "year"]
    
    if not value_cols:
        logger.warning("No numeric columns to aggregate")
        return df
    
    # Build aggregation dict
    agg_dict = {col: "mean" for col in value_cols}
    
    # Keep first value for non-numeric columns (metadata)
    for col in df.columns:
        if col not in group_cols and col not in agg_dict and col != "year":
            agg_dict[col] = "first"
    
    df_agg = df.groupby(group_cols + ["year"], as_index=False).agg(agg_dict)
    
    logger.info(
        f"Aggregated from {len(df)} to {len(df_agg)} rows (yearly)"
    )
    
    return df_agg


# -----------------------------------------------------------------------------
# Main Processing Pipeline
# -----------------------------------------------------------------------------

def process_eurostat_dataset(
    df: pd.DataFrame,
    config: DatasetConfig,
) -> pd.DataFrame:
    """
    Process a single Eurostat dataset through Phases 1-3.
    """
    logger.info(f"Processing Eurostat dataset: {config.indicator_name}")
    logger.info(f"  Input shape: {df.shape}")
    
    # Phase 1: Column cleanup
    df = cleanup_eurostat_columns(df)
    df = standardize_column_names(df, config)
    logger.info(f"  After column cleanup: {df.shape}")
    
    # Phase 2: Geographic filtering
    if config.dataset_type == DatasetType.EUROSTAT_CITY:
        df = filter_eurostat_cities(df)
        df = add_city_name_column(df, config.dataset_type)
    else:
        df = filter_eurostat_countries(df)
    
    df = add_country_columns(df, config.dataset_type)
    logger.info(f"  After geographic filtering: {df.shape}")
    
    # Phase 3: Temporal standardization
    df = standardize_time_period(df)
    logger.info(f"  After temporal standardization: {df.shape}")
    
    # Add indicator metadata
    df["indicator"] = config.indicator_name
    df["indicator_desc"] = config.indicator_description
    
    return df


def process_who_dataset(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Process WHO air quality dataset through Phases 1-3.
    
    WHO data has a different structure with multiple pollutant columns.
    """
    logger.info("Processing WHO air quality dataset")
    logger.info(f"  Input shape: {df.shape}")
    
    # Phase 1: Column standardization (WHO has different structure)
    df = df.rename(columns={
        "city": "location_code",
        "year": "time_period",
    })
    
    # Keep only relevant columns
    keep_cols = [
        "location_code", "time_period", "iso3", "country_name",
        "pm10_concentration", "pm25_concentration", "no2_concentration",
        "population", "latitude", "longitude",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    logger.info(f"  After column cleanup: {df.shape}")
    
    # Phase 2: Geographic filtering
    df = filter_who_cities(df)
    df = add_city_name_column(df, DatasetType.WHO_AIR_QUALITY)
    df = add_country_columns(df, DatasetType.WHO_AIR_QUALITY)
    logger.info(f"  After geographic filtering: {df.shape}")
    
    # Phase 3: Temporal standardization
    df = standardize_time_period(df)
    logger.info(f"  After temporal standardization: {df.shape}")
    
    return df


def load_and_process_dataset(
    raw_dir: Path,
    config: DatasetConfig,
) -> Optional[pd.DataFrame]:
    """
    Load a raw CSV and process it through Phases 1-3.
    """
    file_path = raw_dir / config.filename
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    logger.info(f"Loading: {file_path.name}")
    df = pd.read_csv(file_path, low_memory=False)
    
    if config.dataset_type == DatasetType.WHO_AIR_QUALITY:
        return process_who_dataset(df, config)
    else:
        return process_eurostat_dataset(df, config)


def generate_coverage_report(
    processed_datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Generate a data coverage report showing availability per city/year.
    """
    coverage_records = []
    
    for dataset_name, df in processed_datasets.items():
        if df is None or df.empty:
            continue
        
        # Get unique city-year combinations
        if "city_name" in df.columns:
            location_col = "city_name"
        elif "country_code" in df.columns:
            location_col = "country_code"
        else:
            continue
        
        for location in df[location_col].dropna().unique():
            location_df = df[df[location_col] == location]
            years_available = sorted(location_df["year"].dropna().unique())
            
            coverage_records.append({
                "dataset": dataset_name,
                "location": location,
                "n_years": len(years_available),
                "year_min": min(years_available) if years_available else None,
                "year_max": max(years_available) if years_available else None,
                "years": years_available,
            })
    
    return pd.DataFrame(coverage_records)


def run_preprocessing_pipeline(
    raw_dir: Path,
    output_dir: Path,
    datasets_to_process: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the full preprocessing pipeline (Phases 1-3) on all datasets.
    
    Args:
        raw_dir: Path to raw data directory
        output_dir: Path to output directory for processed files
        datasets_to_process: Optional list of dataset keys to process.
                           If None, processes all configured datasets.
    
    Returns:
        Dictionary of processed DataFrames keyed by dataset name.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which datasets to process
    if datasets_to_process is None:
        configs_to_process = DATASET_CONFIGS
    else:
        configs_to_process = {
            k: v for k, v in DATASET_CONFIGS.items() 
            if k in datasets_to_process
        }
    
    logger.info(f"Processing {len(configs_to_process)} datasets")
    logger.info(f"Target year range: {YEAR_MIN}-{YEAR_MAX}")
    logger.info(f"Target cities: {[c.name for c in TARGET_CITIES]}")
    
    processed_datasets: dict[str, pd.DataFrame] = {}
    
    for name, config in configs_to_process.items():
        try:
            df = load_and_process_dataset(raw_dir, config)
            if df is not None and not df.empty:
                processed_datasets[name] = df
                
                # Save processed dataset
                output_path = output_dir / f"{name}_processed.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Saved: {output_path.name} ({len(df)} rows)")
            else:
                logger.warning(f"No data after processing: {name}")
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            raise
    
    # Generate and save coverage report
    if processed_datasets:
        coverage_df = generate_coverage_report(processed_datasets)
        coverage_path = output_dir / "data_coverage_report.csv"
        coverage_df.to_csv(coverage_path, index=False)
        logger.info(f"Saved coverage report: {coverage_path.name}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Datasets processed: {len(processed_datasets)}")
        print(f"Year range: {YEAR_MIN}-{YEAR_MAX}")
        print("\nRows per dataset:")
        for name, df in processed_datasets.items():
            print(f"  {name}: {len(df):,} rows")
        print("=" * 60)
    
    return processed_datasets


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess European city data (Phases 1-3)"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "raw",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "output" / "processed",
        help="Path to output directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="Specific datasets to process (default: all)",
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
    
    run_preprocessing_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        datasets_to_process=args.datasets,
    )
