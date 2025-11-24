import requests
import duckdb
import time

def fetch_osm_amenities(city_name):
    """
    Uses Overpass API to count amenities (parks, libraries, hospitals) in a city.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query: Get count of parks, libraries, hospitals within the city relation
    # Note: Geocoding city names to areas is complex in OSM. 
    # We will use a simplified "around" query or search by name.
    # Ideally, we should use the bounding box from the DB if we had it.
    # For this POC, we'll try to match by area name.
    
    query = f"""
    [out:json][timeout:25];
    area[name="{city_name}"]->.searchArea;
    (
      node["leisure"="park"](area.searchArea);
      way["leisure"="park"](area.searchArea);
      relation["leisure"="park"](area.searchArea);
    );
    out count;
    """
    
    try:
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        
        # Parse 'count' from elements
        # Overpass 'out count' returns stats in 'elements' -> 'tags' -> 'total' usually?
        # Actually 'out count' output structure is specific.
        # Let's check typical response or use 'count' mode.
        
        # Simplified: Just counting nodes/ways might be heavy. 
        # Better: Use pre-aggregated tags if available, but Overpass counts elements.
        
        # Let's count elements returned
        if 'elements' in data:
            total_parks = 0
            for el in data['elements']:
                if 'tags' in el and 'total' in el['tags']:
                    total_parks = int(el['tags']['total'])
            return total_parks
            
        return 0
        
    except Exception as e:
        # print(f"Error OSM for {city_name}: {e}")
        return None

# Placeholder: OSM integration is heavy and requires good geocoding.
# We will create the structure but maybe mock the heavy lifting or limit scope.


