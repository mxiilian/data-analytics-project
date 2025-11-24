# Data Documentation (`data.md`)

## 1. Datasets Overview

### A. Eurostat Urban Audit & Related Datasets (CSV)
These files follow the Eurostat SDMX linear format and have been fully normalized.

| Filename | Domain | Dataset Name / Description | Key Indicators |
| :--- | :--- | :--- | :--- |
| `urb_cpop1` | Demographics | Population on 1 January | Total population, Age groups |
| `urb_ctran` | Transport | Transport in cities | Registered cars, Public transport share |
| `hlth_rs_bds1`| Health | Hospital beds | Beds per 100k inhabitants |
| `sdg_11_60` | Sustainability | Recycling rate of municipal waste | Recycling % |
| `sdg_17_60` | Digital | High-speed internet coverage | % households with VHCN (Fiber/5G) |
| `tour_occ_nim`| Tourism | Tourist accommodation nights (Monthly) | Hotel/Camping nights |
| `urb_ceduc` | Education | Education in cities | Students in higher education |
| `urb_clma` | Economy/Labor | Labour market in cities | Unemployment, Activity rates |
| `urb_cpopcb` | Demographics | Population by citizenship | Foreign-born population |
| `urb_ctour` | Tourism | Culture and tourism (Annual) | Total nights spent |
| `isoc_ciegi_ac`| Digital/Gov | E-government activities | Internet interaction with public authorities |
| `trng_lfse_01`| Education | Adult Education Participation | Participation rate in training (last 4 weeks) |

### B. Excel Datasets
These files contain supplemental data and are processed generically.

| Filename | Content | Format |
| :--- | :--- | :--- |
| `who_ambient_air_quality...xlsx` | WHO Air Quality Database | PM10/PM2.5 concentrations by city/year |
| `quality_of_life_european...xlsx` | Quality of Life Survey 2023 | Subjective satisfaction scores (0-10 or %) |
| `Share_of_green_areas...xlsx` | Green Areas | Share of green urban areas (%) |
| `Daten_DA.xlsx` | Miscellaneous | Custom dataset (likely aggregating specific metrics) |

---

## 2. Transformation Logic

The raw data is transformed into a **Star Schema** to enable easy cross-referencing of cities and years.

1.  **Normalization**:
    *   **Geography**: All city/country codes (`AT`, `DE1001V`) are extracted into a unique `dim_geo` table.
    *   **Time**: All years/months are standardized into `dim_time` (integers).
    *   **Indicators**: We separate the *definition* of the data from the *value*.
        *   Raw CSV columns like `unit`, `sex`, `age` are combined to create a unique `indicator_code`.
        *   Descriptions (e.g., "Hospital beds") are stored in `indicator_name`.

2.  **Handling Metadata**:
    *   The ETL pipeline automatically pairs "Code" columns with their "Description" columns (e.g., `HBEDT` -> "Available beds in hospitals").
    *   This ensures you don't just see `I_IUGOV12` but also "Internet use: interaction with public authorities".

3.  **Excel Unpivoting**:
    *   "Wide" tables (years/indicators as columns) are "melted" into a "Long" format (Geo, Year, Indicator, Value).

---

## 3. Analysis Guide: Creating "Smart City" Scores

To calculate composite scores like "Smart Economy" or "Quality of Life" (QoL), you will need to **pivot** the data back into a wide format where each city-year is a row and selected indicators are columns.

### Step 1: Identify Relevant Indicators
First, explore `dim_indicator` to find the exact codes for your metrics.

```sql
SELECT indicator_code, indicator_name, unit 
FROM dim_indicator 
WHERE domain IN ('URB', 'SDG', 'ISOC') 
  AND (indicator_name LIKE '%internet%' OR indicator_name LIKE '%transport%');
```

**Example "Smart Economy" Indicators:**
-   **Digital**: `SDG_17_60_...` (High-speed internet coverage)
-   **Innovation**: `ISOC_CIEGI_AC_...` (E-gov interaction)
-   **Education**: `URB_CEDUC_TE1026V` (Students in higher ed)
-   **Labor**: `URB_CLMA_EC1010V` (Unemployment - *Negative factor*)

**Example "QoL" Indicators:**
-   **Environment**: `SDG_11_60_...` (Recycling rate), Green Areas (Excel)
-   **Health**: `HLTH_RS_BDS1_...` (Hospital beds)
-   **Safety/Transport**: `URB_CTRAN_TT1057I` (Cars per capita - *contextual*)

### Step 2: Aggregation & Normalization (Z-Scores)
Since units differ (%, raw counts, per 100k), you must normalize values before averaging.

**Python Workflow:**
```python
import duckdb
import pandas as pd
from sklearn.preprocessing import StandardScaler

con = duckdb.connect('data.duckdb')

# 1. Fetch Data for selected indicators
query = """
    SELECT g.geo_name, f.year, i.domain, i.indicator_name, f.value
    FROM fact_measurements f
    JOIN dim_indicator i ON f.indicator_code = i.indicator_code
    JOIN dim_geo g ON f.geo_code = g.geo_code
    WHERE i.indicator_code IN ('YOUR_SELECTED_CODES_HERE')
    AND f.year = 2020
"""
df = con.execute(query).df()

# 2. Pivot
df_pivot = df.pivot_table(index='geo_name', columns='indicator_name', values='value')

# 3. Normalize (Z-Score)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_pivot), index=df_pivot.index, columns=df_pivot.columns)

# 4. Calculate Score
# Simple average of scaled values (invert sign for negative metrics like unemployment)
df_scaled['Smart_Economy_Score'] = df_scaled[['Internet...', 'Education...']].mean(axis=1)
```

### Step 3: Handling Missing Data
Urban data is often sparse.
-   **Interpolation**: If 2020 is missing, check 2019 or 2021.
-   **Filtering**: Only score cities that have at least 70% of the required indicators available.
