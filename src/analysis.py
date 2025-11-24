import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calculate_scores(db_path='data/db/data.duckdb'):
    con = duckdb.connect(db_path)
    
    print("--- Loading Data for Analysis ---")
    
    # 1. Select Key Indicators
    # We need to map our concept (e.g. "Internet") to actual codes in DB
    # For this script, we use the codes we found in CSV exploration.
    # Note: You might need to adjust these based on exact codes in your DB (check dim_indicator)
    
    indicators = {
        'Internet': 'SDG_17_60_VHCN_FX', # Example Code
        'E-Gov': 'ISOC_CIEGI_AC_I_IUGOV12',
        'Education': 'URB_CEDUC_TE1026V',
        'Unemployment': 'URB_CLMA_EC1010V',
        'Recycling': 'SDG_11_60',
        'HospitalBeds': 'HLTH_RS_BDS1_HBEDT',
        'AQI': 'API_AQICN_AQI' # Hypothetical
    }
    
    # Fetch Data (Long Format)
    # We try to get the most recent data per city (e.g. max year)
    query = """
    SELECT 
        g.geo_name,
        g.country_code,
        i.indicator_code,
        f.value,
        f.year
    FROM fact_measurements f
    JOIN dim_indicator i ON f.indicator_code LIKE i.indicator_code || '%'
    JOIN dim_geo g ON f.geo_code = g.geo_code
    WHERE 
       i.indicator_code LIKE '%SDG_17_60%' OR
       i.indicator_code LIKE '%ISOC_CIEGI_AC%' OR
       i.indicator_code LIKE '%URB_CLMA%' OR
       i.indicator_code LIKE '%HLTH_RS_BDS1%'
    """
    
    df = con.execute(query).df()
    
    if df.empty:
        print("No data found for scoring. Check indicator codes.")
        return

    # 2. Preprocessing
    # Pivot: Index=City, Col=Indicator
    # We take the mean value per city/indicator (averaging over years if multiple recent years)
    # Ideally, filter for year > 2018
    df_recent = df[df['year'] >= 2018]
    
    # We need to simplify indicator codes to generic names for pivoting
    # This mapping depends on exact strings in your DB
    def map_ind(code):
        if 'SDG_17_60' in code: return 'Internet'
        if 'ISOC_CIEGI' in code: return 'E-Gov'
        if 'URB_CLMA' in code: return 'Unemployment'
        if 'HLTH_RS' in code: return 'HospitalBeds'
        return 'Other'
        
    df_recent['Category'] = df_recent['indicator_code'].apply(map_ind)
    
    # Pivot
    df_pivot = df_recent.pivot_table(index=['geo_name', 'country_code'], columns='Category', values='value', aggfunc='mean')
    
    print(f"Data available for {len(df_pivot)} cities.")
    
    # 3. Imputation (Fill with Country Mean)
    # Group by country_code level
    # Reset index to access country_code
    df_pivot = df_pivot.reset_index()
    
    numeric_cols = df_pivot.select_dtypes(include=np.number).columns
    
    # Fill NA with country mean
    df_pivot[numeric_cols] = df_pivot.groupby('country_code')[numeric_cols].transform(lambda x: x.fillna(x.mean()))
    
    # Fill remaining (if country has no data) with global mean
    df_pivot[numeric_cols] = df_pivot[numeric_cols].fillna(df_pivot[numeric_cols].mean())
    
    # 4. Calculation
    # Normalize
    scaler = MinMaxScaler()
    df_scaled = df_pivot.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_pivot[numeric_cols])
    
    # Scores
    # Weights
    # Smart Economy: Internet (0.5) + E-Gov (0.5) - Unemployment (0.5)
    # Note: Unemployment is negative, so we subtract or invert.
    
    if 'Internet' in df_scaled and 'E-Gov' in df_scaled:
        df_scaled['Smart_Score'] = (
            0.4 * df_scaled.get('Internet', 0) + 
            0.4 * df_scaled.get('E-Gov', 0) - 
            0.2 * df_scaled.get('Unemployment', 0)
        )
    else:
        df_scaled['Smart_Score'] = 0
        
    # QoL: HospitalBeds (0.5) + ...
    df_scaled['QoL_Score'] = (
        0.5 * df_scaled.get('HospitalBeds', 0)
        # Add AQI etc here
    )
    
    # 5. Output
    results = df_scaled[['geo_name', 'Smart_Score', 'QoL_Score']].sort_values('QoL_Score', ascending=False)
    print("\nTop 10 Cities by QoL:")
    print(results.head(10))
    
    results.to_csv('output/qol_scores.csv', index=False)
    print("\nScores saved to output/qol_scores.csv")

if __name__ == "__main__":
    calculate_scores()


