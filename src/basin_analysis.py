"""
basin_analysis.py - Major Indian river basin masks and rainfall accumulation.
Basin boundaries from GRDC (Global Runoff Data Centre) approximate coordinates.
"""
import numpy as np
import warnings

# Major Indian river basins: simplified bounding polygon boxes
# Real implementations use GRDC or SWAT shapefiles (we use lat/lon box approximations)
RIVER_BASINS = {
    "Ganga-Yamuna":    {"lat": (24.0, 31.5), "lon": (73.0, 88.5),  "area_km2": 861_000, "color": "#1f77b4"},
    "Brahmaputra":     {"lat": (24.0, 29.5), "lon": (89.5, 97.5),  "area_km2": 580_000, "color": "#2ca02c"},
    "Indus":           {"lat": (24.5, 36.5), "lon": (66.5, 78.0),  "area_km2": 321_000, "color": "#9467bd"},
    "Godavari":        {"lat": (16.0, 22.0), "lon": (73.5, 82.5),  "area_km2": 312_000, "color": "#8c564b"},
    "Krishna":         {"lat": (13.5, 19.5), "lon": (73.5, 81.5),  "area_km2": 258_000, "color": "#e377c2"},
    "Mahanadi":        {"lat": (19.0, 23.5), "lon": (80.5, 86.5),  "area_km2": 141_000, "color": "#7f7f7f"},
    "Narmada":         {"lat": (21.0, 24.0), "lon": (72.5, 80.5),  "area_km2":  98_000, "color": "#bcbd22"},
    "Cauvery":         {"lat": ( 9.5, 13.5), "lon": (75.0, 79.5),  "area_km2":  81_000, "color": "#17becf"},
}

def compute_basin_rainfall_accumulation(rain_da, days_back=7):
    """
    Compute total accumulated rainfall (mm) over last N days for each major basin.

    Returns list of dicts: {basin_name, total_mm, area_km2, volume_km3, color}
    """
    lats = rain_da.lat.values
    lons = rain_da.lon.values
    n = min(days_back, len(rain_da.time))
    recent = rain_da.values[-n:]   # (days, lat, lon)

    results = []
    for name, info in RIVER_BASINS.items():
        lat_min, lat_max = info["lat"]
        lon_min, lon_max = info["lon"]
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            continue
        basin_rain = recent[:, lat_mask, :][:, :, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_rain) / (lat_mask.sum() * lon_mask.sum()))
        # Volume = area (km2) * depth (mm) * 1e-6 (km3)
        volume_km3 = info["area_km2"] * total_mm * 1e-6
        results.append({
            "basin": name,
            "total_mm": round(total_mm, 1),
            "area_km2": info["area_km2"],
            "volume_km3": round(volume_km3, 2),
            "color": info["color"]
        })
    results.sort(key=lambda x: -x["total_mm"])
    return results

def compute_basin_forecast_accumulation(predictions, rain_da):
    """
    Compute 7-day forecast accumulated rainfall for each major basin.
    predictions: np.ndarray (days, lat, lon)
    """
    lats = rain_da.lat.values
    lons = rain_da.lon.values
    results = []
    for name, info in RIVER_BASINS.items():
        lat_min, lat_max = info["lat"]
        lon_min, lon_max = info["lon"]
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            continue
        basin_pred = predictions[:, lat_mask, :][:, :, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_pred) / (lat_mask.sum() * lon_mask.sum()))
        results.append({
            "basin": name,
            "forecast_total_mm": round(total_mm, 1),
            "area_km2": info["area_km2"],
            "color": info["color"]
        })
    results.sort(key=lambda x: -x["forecast_total_mm"])
    return results
