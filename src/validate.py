import duckdb

def run_validation(db_path='data/db/data.duckdb'):
    print("\n--- STARTING VALIDATION ---")
    try:
        con = duckdb.connect(db_path)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    # 1. Row Counts
    try:
        meas_count = con.execute("SELECT COUNT(*) FROM fact_measurements").fetchone()[0]
        geo_count = con.execute("SELECT COUNT(*) FROM dim_geo").fetchone()[0]
        ind_count = con.execute("SELECT COUNT(*) FROM dim_indicator").fetchone()[0]
        
        print(f"Fact Measurements: {meas_count}")
        print(f"Geographies: {geo_count}")
        print(f"Indicators: {ind_count}")
        
        if meas_count == 0:
            print("WARNING: No measurements found!")
    except Exception as e:
        print(f"Error querying counts: {e}")

    # 2. Top Cities by Data Points
    try:
        print("\nTop 5 Cities by Data Volume:")
        res = con.execute("""
            SELECT g.geo_name, COUNT(*) as cnt 
            FROM fact_measurements f
            JOIN dim_geo g ON f.geo_code = g.geo_code
            GROUP BY g.geo_name
            ORDER BY cnt DESC
            LIMIT 5
        """).fetchall()
        for r in res:
            print(f"  {r[0]}: {r[1]}")
    except: pass

    # 3. Data Integrity Check (Nulls)
    nulls = con.execute("SELECT COUNT(*) FROM fact_measurements WHERE value IS NULL").fetchone()[0]
    print(f"\nNull Values in Measurements: {nulls}")

    con.close()
    print("--- VALIDATION COMPLETE ---\n")

if __name__ == "__main__":
    run_validation()

