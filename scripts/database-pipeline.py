import pandas as pd
import duckdb
import os
import glob

# Configuration
DB_PATH = '../data/db/data-full.duckdb'

# Columns to explicitly exclude from Indicator construction
EXCLUDE_COLS = {
    'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME',
    'freq', 'Time frequency',
    'unit', 'Unit of measure',
    'geo', 'Geopolitical entity (reporting)', 'Geopolitical entity (declaring)', 'cities',
    'TIME_PERIOD', 'Time',
    'OBS_VALUE', 'Observation value',
    'OBS_FLAG', 'Observation status (Flag) V2 structure',
    'CONF_STATUS', 'Confidentiality status (flag)',
    'dataflow',
    'Sex'
}


def get_db_connection(db_path=DB_PATH):
    return duckdb.connect(db_path)


def setup_schema(con):
    """Creates the Star Schema if it doesn't exist."""
    con.execute("DROP TABLE IF EXISTS fact_measurements")
    con.execute("DROP TABLE IF EXISTS dim_geo")
    con.execute("DROP TABLE IF EXISTS dim_time")
    con.execute("DROP TABLE IF EXISTS dim_indicator")

    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_geo (
        geo_code VARCHAR PRIMARY KEY,
        geo_name VARCHAR,
        country_code VARCHAR,
        iso_code VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_time (
        year INTEGER PRIMARY KEY
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_indicator (
        indicator_code VARCHAR PRIMARY KEY,
        indicator_name VARCHAR,
        dataset_name VARCHAR,
        unit VARCHAR,
        domain VARCHAR
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS fact_measurements (
        geo_code VARCHAR,
        year INTEGER,
        indicator_code VARCHAR,
        value DOUBLE,
        flag VARCHAR,
        FOREIGN KEY (geo_code) REFERENCES dim_geo(geo_code),
        FOREIGN KEY (year) REFERENCES dim_time(year),
        FOREIGN KEY (indicator_code) REFERENCES dim_indicator(indicator_code)
    );
    """)
    print("Schema initialized.")


def get_indicator_columns(df_columns):
    """Identifies columns that define the indicator (dimensions)."""
    potential = []
    for c in df_columns:
        if c in EXCLUDE_COLS:
            continue
        if ' ' in c:
            continue
        potential.append(c)
    return potential


def process_csv_file(filepath, con):
    """Verarbeitet Eurostat CSV-Dateien."""
    print(f"Processing CSV:  {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    cols = df.columns.tolist()

    geo_col = next((c for c in cols if c in ['geo', 'cities']), None)
    time_col = 'TIME_PERIOD'
    val_col = 'OBS_VALUE'
    flag_col = 'OBS_FLAG'

    if not geo_col or time_col not in cols or val_col not in cols:
        print(f"Skipping {filepath}: Missing essential columns (geo/time/value).")
        return

    ind_cols = get_indicator_columns(cols)

    code_to_desc = {}
    for c in ind_cols:
        idx = cols.index(c)
        if idx + 1 < len(cols):
            candidate = cols[idx + 1]
            if ' ' in candidate and candidate not in EXCLUDE_COLS:
                code_to_desc[c] = candidate

    unit_col = 'unit' if 'unit' in cols else None
    dataset_id = os.path.basename(filepath).split('_page_')[0].upper()

    structure_name_col = 'STRUCTURE_NAME'
    dataset_desc = "Unknown Dataset"
    if structure_name_col in df.columns:
        dataset_desc = str(df[structure_name_col].iloc[0])

    geo_name_col = next((c for c in cols if 'Geopolitical' in c and c != geo_col), None)

    unique_geos = df[[geo_col]].drop_duplicates()
    if geo_name_col:
        unique_geos = df[[geo_col, geo_name_col]].drop_duplicates()

    for _, row in unique_geos.iterrows():
        code = str(row[geo_col])
        name = row[geo_name_col] if geo_name_col else code
        country = code[:2] if len(code) >= 2 else code
        con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (code, name, country, country))

    df['ind_code'] = dataset_id
    df['ind_desc_parts'] = ""

    for c in ind_cols:
        df['ind_code'] = df['ind_code'] + "_" + df[c].astype(str)
        if c in code_to_desc:
            desc_col = code_to_desc[c]
            df['ind_desc_parts'] = df['ind_desc_parts'] + " - " + df[desc_col].astype(str)
        else:
            df['ind_desc_parts'] = df['ind_desc_parts'] + " - " + df[c].astype(str)

    sel_cols = ['ind_code', 'ind_desc_parts']
    if unit_col:  sel_cols.append(unit_col)

    unique_inds = df[sel_cols].drop_duplicates()

    for _, row in unique_inds.iterrows():
        code = row['ind_code']
        unit = row[unit_col] if unit_col else 'N/A'
        specific_desc = str(row['ind_desc_parts']).strip(" - ")
        if specific_desc:
            full_name = f"{dataset_desc}:  {specific_desc}"
        else:
            full_name = dataset_desc
        domain = dataset_id.split('_')[0]
        con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)",
                    (code, full_name, dataset_desc, unit, domain))

    df['year_clean'] = df[time_col].astype(str).str[:4]
    df = df[df['year_clean'].str.isnumeric()]
    df['year_clean'] = df['year_clean'].astype(int)

    unique_years = df['year_clean'].unique()
    for y in unique_years:
        con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (int(y),))

    fact_df = df[[geo_col, 'year_clean', 'ind_code', val_col, flag_col]].copy()
    fact_df.columns = ['geo_code', 'year', 'indicator_code', 'value', 'flag']
    fact_df = fact_df.dropna(subset=['value'])

    con.register('temp_facts', fact_df)
    con.execute("INSERT INTO fact_measurements SELECT * FROM temp_facts")
    con.unregister('temp_facts')

    print(f"  -> Loaded {len(fact_df)} rows.")


def process_who_air_quality(con):
    """
    Verarbeitet die WHO Ambient Air Quality Datenbank.
    Erstellt separate Indikatoren für PM10, PM2.5 und NO2.
    """
    filepath = "../data/raw/who_ambient_air_quality_database_version_2024_(v6.1).csv"

    if not os.path.exists(filepath):
        print(f"WHO file not found: {filepath}")
        return

    print(f"Processing WHO Air Quality Data:  {filepath}")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading WHO file: {e}")
        return

    # Dataset info
    dataset_name = "WHO Ambient Air Quality Database 2024"
    domain = "WHO"

    # Definiere die Luftqualitäts-Indikatoren
    indicators = {
        'pm10_concentration': {
            'code': 'WHO_PM10',
            'name': 'PM10 Concentration (µg/m³)',
            'unit': 'µg/m³'
        },
        'pm25_concentration': {
            'code': 'WHO_PM25',
            'name': 'PM2.5 Concentration (µg/m³)',
            'unit': 'µg/m³'
        },
        'no2_concentration': {
            'code': 'WHO_NO2',
            'name': 'NO2 Concentration (µg/m³)',
            'unit': 'µg/m³'
        }
    }

    # Registriere Indikatoren
    for col, info in indicators.items():
        con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)",
                    (info['code'], info['name'], dataset_name, info['unit'], domain))

    rows_loaded = 0

    for _, row in df.iterrows():
        try:
            # Erstelle geo_code aus Stadt und Land
            city = str(row['city']).strip()
            country_code = str(row['iso3']).strip()
            country_name = str(row['country_name']).strip()

            # Geo-Code:  Kombiniere ISO3 und Stadt für Eindeutigkeit
            # Bereinige Stadtname für Code (entferne Sonderzeichen)
            city_clean = city.replace('/', '_').replace(' ', '_').replace(',', '')
            geo_code = f"{country_code}_{city_clean}"

            # Kürze geo_code falls zu lang
            if len(geo_code) > 50:
                geo_code = geo_code[:50]

            # Jahr
            year = int(row['year'])

            # Registriere Geo
            con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ? )",
                        (geo_code, f"{city}, {country_name}", country_code, country_code))

            # Registriere Jahr
            con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (year,))

            # Füge Messwerte für jeden Indikator ein
            for col, info in indicators.items():
                value = row.get(col)
                if pd.notna(value) and value != '' and value != 'NA':
                    try:
                        value_float = float(value)
                        con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)",
                                    (geo_code, year, info['code'], value_float, None))
                        rows_loaded += 1
                    except (ValueError, TypeError):
                        continue

        except Exception:
            continue

    print(f"  -> Loaded {rows_loaded} WHO air quality measurements.")


def process_excel_files(con):
    """Verarbeitet Excel-Dateien."""
    files = glob.glob("data/raw/*.xlsx")
    for f in files:
        # Überspringe temporäre Excel-Dateien
        if os.path.basename(f).startswith("~$"):
            continue

        print(f"Processing Excel:  {f}")

        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(f, sheet_name=sheet)
                cols_map = {c: c.lower() for c in df.columns}

                geo_col_orig = next((c for c, lc in cols_map.items() if 'city' in lc or 'geo' in lc or 'country' in lc),
                                    None)
                time_col_orig = next((c for c, lc in cols_map.items() if 'year' in lc or 'time' in lc), None)

                if geo_col_orig and time_col_orig:
                    val_col_orig = next((c for c, lc in cols_map.items() if 'value' in lc or 'score' in lc), None)

                    if val_col_orig:
                        print(f"  Sheet {sheet}:  Detected Long format")
                        ind_code = f"EXCEL_{os.path.basename(f)}_{sheet}".replace(" ", "_").replace(". xlsx",
                                                                                                    "").upper()
                        con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)",
                                    (ind_code, f"{sheet} ({f})", f"{f} - {sheet}", "N/A", "EXCEL"))

                        for _, row in df.iterrows():
                            try:
                                y = int(str(row[time_col_orig])[:4])
                                g = str(row[geo_col_orig])
                                v = row[val_col_orig]

                                con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (g, g, g[: 2], g[: 2]))
                                con.execute("INSERT OR IGNORE INTO dim_time VALUES (? )", (y,))
                                con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)",
                                            (g, y, ind_code, v, None))
                            except:
                                continue
                    else:
                        print(f"  Sheet {sheet}: Detected Wide format")
                        id_vars = [geo_col_orig, time_col_orig]
                        value_vars = [c for c in df.columns if c not in id_vars]
                        melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='indicator',
                                         value_name='value')

                        for _, row in melted.iterrows():
                            try:
                                y = int(str(row[time_col_orig])[:4])
                                g = str(row[geo_col_orig])
                                ind_raw = str(row['indicator'])
                                v = row['value']

                                full_ind_code = f"EXCEL_{os.path.basename(f)}_{sheet}_{ind_raw}".replace(" ",
                                                                                                         "_").replace(
                                    ".xlsx", "").upper()
                                con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)",
                                            (full_ind_code, ind_raw, f"{f} - {sheet}", "N/A", "EXCEL"))
                                con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (g, g, g[:2], g[:2]))
                                con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (y,))
                                con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)",
                                            (g, y, full_ind_code, v, None))
                            except:
                                continue

        except Exception as e:
            print(f"Error processing Excel {f}: {e}")


def run_etl():
    con = get_db_connection()
    setup_schema(con)

    # CSVs - ERWEITERT:  Suche nach _2_0 UND _2_1 Dateien
    csv_patterns = [
        "../data/raw/*_page_linear_2_0.csv",
        "../data/raw/*_page_linear_2_1.csv",  # NEU:  Auch _2_1 Dateien
    ]

    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern))

    # Entferne Duplikate
    csv_files = list(set(csv_files))

    print(f"Found {len(csv_files)} Eurostat CSV files.")
    for f in csv_files:
        process_csv_file(f, con)

    # WHO Air Quality Data - NEU
    process_who_air_quality(con)

    # Excels
    process_excel_files(con)

    print("\nETL Pipeline Completed Successfully.")
    con.close()


if __name__ == "__main__":
    run_etl()