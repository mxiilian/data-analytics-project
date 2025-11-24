import duckdb

def create_schema(db_path='data.duckdb'):
    con = duckdb.connect(db_path)
    
    # Dimension: Geography
    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_geo (
        geo_code VARCHAR PRIMARY KEY,
        geo_name VARCHAR,
        country_code VARCHAR,
        iso_code VARCHAR
    );
    """)
    
    # Dimension: Time
    # We can just use the integer year in the fact table, but a dimension allows for more detail later
    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_time (
        year INTEGER PRIMARY KEY
    );
    """)
    
    # Dimension: Indicator
    con.execute("""
    CREATE TABLE IF NOT EXISTS dim_indicator (
        indicator_code VARCHAR PRIMARY KEY,
        indicator_name VARCHAR,
        unit VARCHAR,
        domain VARCHAR
    );
    """)
    
    # Fact: Measurements
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
    
    print("Schema created successfully.")
    con.close()

if __name__ == "__main__":
    create_schema()

