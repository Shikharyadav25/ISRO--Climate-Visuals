import json
import os
import re

def simplify_polygon(coords, tolerance=0.1):
    # Extremely basic Douglas-Peucker or just skip points
    # Actually, skipping points is bad. Let's just use every Nth point if it's too large,
    # or just use all points since matplotlib Path handles 10000 points instantly!
    # Wait, writing 10,000 points to a Python file will make it 5MB.
    return coords

def process_geojson(geojson_path, target_py_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pilot_regions = {}
    state_polygons = {}
    
    for feature in data['features']:
        name = feature['properties'].get('NAME_1')
        if not name:
            continue
            
        geom = feature['geometry']
        if geom['type'] == 'Polygon':
            polys = [geom['coordinates'][0]]
        elif geom['type'] == 'MultiPolygon':
            polys = [p[0] for p in geom['coordinates']]
        else:
            continue
            
        # Get bounds
        all_lats = []
        all_lons = []
        for poly in polys:
            for lon, lat in poly:
                all_lons.append(lon)
                all_lats.append(lat)
                
        lat_min = min(all_lats) - 0.5
        lat_max = max(all_lats) + 0.5
        lon_min = min(all_lons) - 0.5
        lon_max = max(all_lons) + 0.5
        
        # Datameet uses slightly different names sometimes.
        if name == "Orissa": name = "Odisha"
        if name == "Uttaranchal": name = "Uttarakhand"
        
        pilot_regions[name] = (round(lat_min, 3), round(lat_max, 3), round(lon_min, 3), round(lon_max, 3))
        
        # Store all sub-polygons for matplotlib Path
        state_polygons[name] = polys

    # Special entry for All India
    pilot_regions["All India"] = (6.5, 37.5, 66.5, 97.5)

    # Save to JSON instead of dumping into python file to keep python file clean
    processed = {
        "pilot_regions": pilot_regions,
        "state_polygons": state_polygons
    }
    with open('data/processed/india_spatial_data.json', 'w', encoding='utf-8') as f:
        json.dump(processed, f)

process_geojson('data/india_states.geojson', 'src/spatial_predictions.py')
print("Successfully processed geojson into india_spatial_data.json")
