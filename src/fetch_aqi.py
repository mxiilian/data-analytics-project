import requests
import duckdb
import time

# You can get a free token from https://aqicn.org/data-platform/token/
# For demo purposes, we might need a placeholder or user input.
# We will use 'demo' token which works for 'shanghai' but for others we need a real one.
# USER MUST PROVIDE TOKEN.

def fetch_aqi_data(city_name, token='demo'):
    """
    Fetches real-time AQI data for a city from WAQI API.
    Note: 'demo' token only works for Shanghai.
    """
    url = f"https://api.waqi.info/feed/{city_name}/?token={token}"
    try:
        r = requests.get(url)
        data = r.json()
        if data['status'] == 'ok':
            return data['data']
        else:
            return None
    except Exception as e:
        print(f"Error fetching AQI for {city_name}: {e}")
        return None

def update_db_with_aqi(con, city_map, token):
    """
    Iterates over cities in DB, fetches AQI, and inserts as a new measurement.
    """
    # 1. Get list of cities from DB
    cities = con.execute("SELECT DISTINCT geo_name, geo_code FROM dim_geo").fetchall()
    
    # 2. Define Indicator
    ind_code = "API_AQICN_AQI"
    con.execute("INSERT OR IGNORE INTO dim_indicator VALUES (?, ?, ?, ?, ?)", 
                (ind_code, "Real-time Air Quality Index (AQI)", "WAQI API", "Index", "ENV"))

    print(f"Fetching AQI for {len(cities)} cities...")
    
    current_year = 2025 # Or current date
    
    for city_name, geo_code in cities:
        # Rate limit
        time.sleep(0.5) 
        
        # Heuristic: Clean city name (remove 'greater city', etc.)
        clean_name = city_name.split('(')[0].strip()
        
        print(f"  > {clean_name}...", end="")
        data = fetch_aqi_data(clean_name, token)
        
        if data:
            try:
                aqi = data['aqi']
                # Insert into Fact Table
                # We treat real-time data as '2025' or current year for now
                if str(aqi).isnumeric():
                    con.execute("INSERT INTO fact_measurements VALUES (?, ?, ?, ?, ?)", 
                                (geo_code, current_year, ind_code, float(aqi), "real-time"))
                    print(f" OK ({aqi})")
                else:
                    print(" No numeric AQI")
            except:
                print(" Parse Error")
        else:
            print(" Not Found")

if __name__ == "__main__":
    # Test run
    # token = input("Enter AQICN Token: ")
    pass


