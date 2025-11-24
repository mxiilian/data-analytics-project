import sys
import os
import duckdb

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.etl import run_etl
from src.validate import run_validation

# Import API fetchers
# Note: These are optional and require tokens/internet
try:
    from src.fetch_aqi import update_db_with_aqi
except ImportError:
    update_db_with_aqi = None

def main():
    print("========================================")
    print("   URBAN DATA ANALYTICS PIPELINE")
    print("========================================")
    
    # Step 1: Core ETL (Internal Data)
    print("\n[Step 1] Running Core ETL...")
    run_etl()
    
    # Step 2: External Data (Optional)
    # This is where we would call the APIs
    # For now, we wrap it in a try-block or user prompt
    print("\n[Step 2] External Data Integration (Optional)...")
    try:
        if update_db_with_aqi:
             con = duckdb.connect('data/db/data.duckdb')
             # Check if user wants to run this (it takes time and needs token)
             # update_db_with_aqi(con, [], token='demo') 
             print("  - AQI fetcher loaded. (Uncomment in main.py to run with token)")
             con.close()
    except Exception as e:
        print(f"  - External data integration skipped: {e}")

    # Step 3: Analysis
    print("\n[Step 3] Running QoL Analysis...")
    try:
        from src.analysis import calculate_scores
        calculate_scores()
    except Exception as e:
        print(f"Analysis failed: {e}")

    # Step 4: Validation
    print("\n[Step 4] Validating Data...")
    run_validation()
    
    print("\nProcess Finished.")

if __name__ == "__main__":
    main()
