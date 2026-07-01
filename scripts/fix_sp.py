with open('src/spatial_predictions.py', 'r') as f:
    content = f.read()

import re
match = re.search(r'class SpatialClimatePredictor.*', content, re.DOTALL)
if match:
    class_code = match.group(0)
    new_header = """import numpy as np
import xarray as xr
import json
import os
from matplotlib.path import Path

# Dynamically load the absolute geographic polygons and bounding boxes
_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'india_spatial_data.json')
try:
    with open(_data_path, 'r', encoding='utf-8') as f:
        _geo_data = json.load(f)
    PILOT_REGIONS = _geo_data['pilot_regions']
    STATE_POLYGONS = _geo_data['state_polygons']
    
    # Precompute Matplotlib Paths for ultra-fast point-in-polygon masking
    STATE_PATHS = {}
    for state, polys in STATE_POLYGONS.items():
        paths = []
        for poly in polys:
            if len(poly) > 2:
                paths.append(Path(poly))
        STATE_PATHS[state] = paths
except Exception as e:
    print(f"Warning: Failed to load dynamic geospatial data: {e}")
    PILOT_REGIONS = {"All India": (6.5, 37.5, 66.5, 97.5)}
    STATE_POLYGONS = {}
    STATE_PATHS = {}

"""
    with open('src/spatial_predictions.py', 'w') as f:
        f.write(new_header + class_code)
    print('Fixed spatial_predictions.py successfully')
else:
    print('Failed to find class')
