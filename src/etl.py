import pandas as pd
import duckdb
import os
import glob

# Configuration
DB_PATH = 'data/db/data.duckdb'

# Columns to explicitly exclude from Indicator construction
# (Standard columns + Metadata + Known descriptions that might slip through heuristics)
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
    'Sex' # Explicitly exclude "Sex" (Title case) if it appears as a column header
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
    """
    Identifies columns that define the indicator (dimensions).
    Heuristic:
    1. Not in EXCLUDE_COLS.
    2. Header does not contain spaces (Description columns usually have spaces).
    3. Header is not Title Case (unless it's a known code). 
       (Actually, Eurostat codes are usually lowercase or acronyms. Descriptions are Sentence case.)
       We'll stick to: Exclude if header has space.
    """
    potential = []
    for c in df_columns:
        if c in EXCLUDE_COLS:
            continue
        if ' ' in c: # Description column
            continue
        potential.append(c)
    return potential

def process_csv_file(filepath, con):
    print(f"Processing CSV: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    cols = df.columns.tolist()
    
    # Identify functional columns
    geo_col = next((c for c in cols if c in ['geo', 'cities']), None)
    time_col = 'TIME_PERIOD'
    val_col = 'OBS_VALUE'
    flag_col = 'OBS_FLAG'
    
    if not geo_col or time_col not in cols or val_col not in cols:
        print(f"Skipping {filepath}: Missing essential columns (geo/time/value).")
        return

    # Identify Indicator Dimension columns
    ind_cols = get_indicator_columns(cols)
    
    # Map Code Cols to Description Cols
    # Heuristic: Description column is often immediately after the code column
    # or has the code column name as a prefix/substring, or just by position.
    # Given the Eurostat format: Code, Label. 
    # Let's verify by checking if the next column has spaces.
    code_to_desc = {}
    for c in ind_cols:
        idx = cols.index(c)
        if idx + 1 < len(cols):
            candidate = cols[idx+1]
            if ' ' in candidate and candidate not in EXCLUDE_COLS:
                 code_to_desc[c] = candidate
    
    # Extract Unit if available
    unit_col = 'unit' if 'unit' in cols else None
    
    # Dataset ID from filename
    dataset_id = os.path.basename(filepath).split('_page_')[0].upper()
    
    # Get Structure Name (Dataset Description)
    structure_name_col = 'STRUCTURE_NAME'
    dataset_desc = "Unknown Dataset"
    if structure_name_col in df.columns:
        # Assume constant for the file
        dataset_desc = str(df[structure_name_col].iloc[0])

    # 1. GEOMETRY DIMENSION
    # Name column is usually next to geo column or has 'Geopolitical' in it
    geo_name_col = next((c for c in cols if 'Geopolitical' in c and c != geo_col), None)
    
    unique_geos = df[[geo_col]].drop_duplicates()
    if geo_name_col:
        unique_geos = df[[geo_col, geo_name_col]].drop_duplicates()
    
    # Batch insert Geos
    # We use executemany or append. DuckDB is fast with appends.
    for _, row in unique_geos.iterrows():
        code = str(row[geo_col])
        name = row[geo_name_col] if geo_name_col else code
        country = code[:2] if len(code) >= 2 else code
        con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (code, name, country, country))

    # 2. INDICATOR DIMENSION
    # Construct Indicator Code and Name
    df['ind_code'] = dataset_id
    df['ind_desc_parts'] = ""
    
    for c in ind_cols:
        # Update Code
        df['ind_code'] = df['ind_code'] + "_" + df[c].astype(str)
        
        # Update Description
        if c in code_to_desc:
            desc_col = code_to_desc[c]
            # If we have a description, append it: " - [Description]"
            df['ind_desc_parts'] = df['ind_desc_parts'] + " - " + df[desc_col].astype(str)
        else:
             # Fallback to code if no description found
             df['ind_desc_parts'] = df['ind_desc_parts'] + " - " + df[c].astype(str)
    
    # Insert Indicators
    # Select columns needed for distinct indicators
    sel_cols = ['ind_code', 'ind_desc_parts']
    if unit_col: sel_cols.append(unit_col)
    
    unique_inds = df[sel_cols].drop_duplicates()
    
    for _, row in unique_inds.iterrows():
        code = row['ind_code']
        unit = row[unit_col] if unit_col else 'N/A'
        
        # Construct full name: Dataset Name + specific dimensions
        # strip leading " - "
        specific_desc = str(row['ind_desc_parts']).strip(" - ")
        if specific_desc:
            full_name = f"{dataset_desc}: {specific_desc}"
        else:
            full_name = dataset_desc
            
        domain = dataset_id.split('_')[0]
        con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)", (code, full_name, dataset_desc, unit, domain))

    # 3. FACTS
    # Clean Time
    df['year_clean'] = df[time_col].astype(str).str[:4]
    df = df[df['year_clean'].str.isnumeric()]
    df['year_clean'] = df['year_clean'].astype(int)
    
    # Insert Time
    unique_years = df['year_clean'].unique()
    for y in unique_years:
        con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (int(y),))
        
    # Insert Facts
    # Prepare final DF
    fact_df = df[[geo_col, 'year_clean', 'ind_code', val_col, flag_col]].copy()
    fact_df.columns = ['geo_code', 'year', 'indicator_code', 'value', 'flag']
    fact_df = fact_df.dropna(subset=['value']) # Remove empty values
    
    # DuckDB direct insert from DF
    con.register('temp_facts', fact_df)
    con.execute("INSERT INTO fact_measurements SELECT * FROM temp_facts")
    con.unregister('temp_facts')
    
    print(f"  -> Loaded {len(fact_df)} rows.")

def process_excel_files(con):
    files = glob.glob("data/raw/*.xlsx")
    for f in files:
        if f.startswith("~$"): continue
        print(f"Processing Excel: {f}")
        
        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(f, sheet_name=sheet)
                # Normalize columns to lower case for checking
                cols_map = {c: c.lower() for c in df.columns}
                lower_cols = list(cols_map.values())
                
                # Detect Geo/Time
                geo_col_orig = next((c for c, lc in cols_map.items() if 'city' in lc or 'geo' in lc or 'country' in lc), None)
                time_col_orig = next((c for c, lc in cols_map.items() if 'year' in lc or 'time' in lc), None)
                
                if geo_col_orig and time_col_orig:
                    val_col_orig = next((c for c, lc in cols_map.items() if 'value' in lc or 'score' in lc), None)
                    
                    if val_col_orig:
                        # LONG FORMAT
                        print(f"  Sheet {sheet}: Detected Long format")
                        ind_code = f"EXCEL_{os.path.basename(f)}_{sheet}".replace(" ", "_").replace(".xlsx","").upper()
                        # Use sheet name as description for Excel
                        con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)", (ind_code, f"{sheet} ({f})", f"{f} - {sheet}", "N/A", "EXCEL"))
                        
                        for _, row in df.iterrows():
                            try:
                                y = int(str(row[time_col_orig])[:4])
                                g = str(row[geo_col_orig])
                                v = row[val_col_orig]
                                
                                con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (g, g, g[:2], g[:2]))
                                con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (y,))
                                con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)", (g, y, ind_code, v, None))
                            except: continue
                            
                    else:
                        # WIDE FORMAT (Indicators or Years as columns)
                        # Assume columns that are not geo/time are indicators
                        print(f"  Sheet {sheet}: Detected Wide format")
                        id_vars = [geo_col_orig, time_col_orig]
                        value_vars = [c for c in df.columns if c not in id_vars]
                        
                        melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='indicator', value_name='value')
                        
                        for _, row in melted.iterrows():
                            try:
                                y = int(str(row[time_col_orig])[:4])
                                g = str(row[geo_col_orig])
                                ind_raw = str(row['indicator'])
                                v = row['value']
                                
                                full_ind_code = f"EXCEL_{os.path.basename(f)}_{sheet}_{ind_raw}".replace(" ", "_").replace(".xlsx","").upper()
                                con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)", (full_ind_code, ind_raw, f"{f} - {sheet}", "N/A", "EXCEL"))
                                con.execute("INSERT OR IGNORE INTO dim_geo VALUES (?, ?, ?, ?)", (g, g, g[:2], g[:2]))
                                con.execute("INSERT OR IGNORE INTO dim_time VALUES (?)", (y,))
                                con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)", (g, y, full_ind_code, v, None))
                            except: continue

        except Exception as e:
            print(f"Error processing Excel {f}: {e}")

def run_etl():
    con = get_db_connection()
    setup_schema(con)
    
    # CSVs
    csv_files = glob.glob("data/raw/*_page_linear_2_0.csv")
    print(f"Found {len(csv_files)} CSV files.")
    for f in csv_files:
        process_csv_file(f, con)
        
    # Excels
    process_excel_files(con)
    
    print("ETL Pipeline Completed Successfully.")
    con.close()

