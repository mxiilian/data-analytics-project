# Urban Data Analytics Project

This project processes urban data from Eurostat (CSV) and other sources (Excel) into a unified DuckDB database for analysis, enabling Quality of Life (QoL) and Smart Economy scoring.

## 📂 Project Structure

```
├── main.py                # Entry point for the pipeline
├── requirements.txt       # Python dependencies
├── src/
│   ├── etl.py             # ETL logic (CSV/Excel -> DuckDB)
│   ├── validate.py        # Data validation scripts
│   ├── analysis.py        # QoL and Smart Economy scoring
│   ├── fetch_aqi.py       # (Optional) External API Integration for Air Quality
│   └── fetch_osm.py       # (Optional) External API Integration for Amenities
├── data/
│   ├── raw/               # Source files (*.csv, *.xlsx)
│   └── db/                # Generated DuckDB database (data.duckdb)
├── docs/
│   ├── data.md            # Detailed dataset documentation & dictionary
│   └── qol_plan.md        # Analysis planning document
├── output/                # Generated reports and scores
│   └── qol_scores.csv
└── scripts/               # Helper utilities
    ├── create_schema.py
    └── inspect_excel_structure.py
```

## 🚀 Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline**:
    ```bash
    python3 main.py
    ```
    This command will:
    -   Initialize the database in `data/db/data.duckdb`.
    -   Process all files in `data/raw/`.
    -   Run validation checks.
    -   Calculate QoL scores and save them to `output/qol_scores.csv`.

## 📊 Database Schema

The data is normalized into a **Star Schema** within DuckDB. See `docs/data.md` for full documentation on indicators and data lineage.

-   **`dim_geo`**: Cities and Countries.
-   **`dim_time`**: Year dimension.
-   **`dim_indicator`**: Metadata for ~12 Eurostat datasets + Excel sources.
-   **`fact_measurements`**: Unified table for all data points.

## 🔍 Analysis

You can connect to the database using DBeaver or Python:

```python
import duckdb
con = duckdb.connect('data/db/data.duckdb')
df = con.execute("SELECT * FROM fact_measurements LIMIT 10").df()
```

For methodology on the QoL scores, refer to `docs/data.md`.
