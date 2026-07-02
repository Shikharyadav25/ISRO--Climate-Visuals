"""
climate_indices.py - NASA/ISRO-grade climate diagnostic indices.
SPI, Flash Flood Guidance, Crop Water Stress, Monsoon Onset, Probabilistic Maps, Brier Skill Score
"""
import numpy as np
import warnings

# SPI classification thresholds (WMO-No. 1090)
SPI_CATEGORIES = [
    (2.0,  float('inf'), "Extremely Wet",     "#08306b"),
    (1.5,  2.0,          "Very Wet",           "#2171b5"),
    (1.0,  1.5,          "Moderately Wet",     "#6baed6"),
    (-1.0, 1.0,          "Near Normal",        "#41ab5d"),
    (-1.5, -1.0,         "Moderately Dry",     "#fe9929"),
    (-2.0, -1.5,         "Severely Dry",       "#d94801"),
    (float('-inf'), -2.0,"Extremely Dry",      "#7f0000"),
]

def compute_spi_spatial(rain_da, clim_atlas, window_days=30):
    """
    Gridded SPI. SPI = (P_obs - P_clim_mean) / P_clim_std
    window_days=30 -> SPI-1, window_days=90 -> SPI-3
    Returns: np.ndarray (lat, lon)
    """
    import pandas as pd
    times = pd.to_datetime(rain_da.time.values)
    n = min(window_days, len(times))
    obs_acc = np.nansum(rain_da.values[-n:], axis=0)

    last_date = times[-1]
    clim_daily_grids = []
    for offset in range(n):
        d = last_date - pd.Timedelta(days=offset)
        key = (d.month, d.day)
        g = clim_atlas.get(key)
        if g is not None:
            clim_daily_grids.append(np.nan_to_num(g, nan=0.0))

    if not clim_daily_grids:
        return np.zeros_like(obs_acc)

    clim_stack = np.array(clim_daily_grids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clim_mean_daily = np.nanmean(clim_stack, axis=0)
        clim_std_daily  = np.nanstd(clim_stack, axis=0)

    clim_mean_acc = clim_mean_daily * n
    clim_std_acc  = clim_std_daily  * np.sqrt(n)
    clim_std_acc  = np.where(clim_std_acc < 0.1, 0.1, clim_std_acc)

    spi = (obs_acc - clim_mean_acc) / clim_std_acc
    spi = np.clip(spi, -4.0, 4.0)
    nan_mask = np.isnan(rain_da.values[-1])
    spi[nan_mask] = np.nan
    return spi

def spi_category(value):
    if np.isnan(value): return "No Data", "#888888"
    for lo, hi, label, color in SPI_CATEGORIES:
        if lo <= value < hi: return label, color
    return "Near Normal", "#41ab5d"

def compute_flash_flood_guidance(rain_da, soil_moisture_grid):
    """
    Flash Flood Guidance risk index. Range 0-5.
    Based on NOAA NWS operational FFG concept.
    """
    rain_intensity = np.nan_to_num(rain_da.values[-1], nan=0.0)
    sm = np.clip(np.nan_to_num(soil_moisture_grid, nan=50.0), 0, 100)
    field_capacity_deficit = (100.0 - sm) * 0.3
    ffg_threshold = np.maximum(5.0, field_capacity_deficit)
    ffg_risk = rain_intensity / ffg_threshold
    ffg_risk = np.clip(ffg_risk, 0, 5.0)
    nan_mask = np.isnan(rain_da.values[-1])
    ffg_risk[nan_mask] = np.nan
    return ffg_risk

def ffg_category(value):
    if np.isnan(value): return "No Data", "#888888"
    if value < 0.5:  return "Low Risk",             "#1a9641"
    if value < 1.0:  return "Moderate (Watch)",     "#fdae61"
    if value < 1.5:  return "High (Warning)",       "#d7191c"
    return "Extreme (Emergency)", "#7f0000"

def compute_crop_water_stress(rain_da, temp_da):
    """
    FAO-56 Crop Water Stress Index. Range 0 (no stress) to 1 (wilting).
    """
    rain_30d = np.nanmean(rain_da.values[-30:], axis=0) if len(rain_da.time) >= 30 \
               else np.nanmean(rain_da.values, axis=0)
    rain_30d = np.nan_to_num(rain_30d, nan=0.0)
    water_avail = 1.0 - np.exp(-rain_30d / 10.0)
    et_ref = 5.0
    et_actual = water_avail * et_ref

    temp_30d = np.nanmean(temp_da.values[-30:], axis=0) if len(temp_da.time) >= 30 \
               else np.nanmean(temp_da.values, axis=0)
    temp_30d = np.nan_to_num(temp_30d, nan=30.0)
    et_crop = np.clip(0.0023 * (temp_30d + 17.8) * 30.0 * 0.408, 2.0, 12.0)

    cwsi = np.clip(1.0 - (et_actual / et_crop), 0.0, 1.0)
    nan_mask = np.isnan(rain_da.values[-1])
    cwsi[nan_mask] = np.nan
    return cwsi

def cwsi_category(value):
    if np.isnan(value): return "No Data", "#888888"
    if value < 0.1:  return "No Stress",              "#1a9641"
    if value < 0.3:  return "Mild Stress",             "#a6d96a"
    if value < 0.5:  return "Moderate Stress",         "#fdae61"
    if value < 0.7:  return "Severe Stress",           "#d7191c"
    return "Extreme Stress (Wilting)", "#7f0000"

MONSOON_WAYPOINTS = [
    ("Kerala",          ( 8.0, 12.0, 74.0, 77.5), 152, 2.5),
    ("Karnataka",       (12.0, 16.0, 74.0, 78.0), 160, 2.5),
    ("Goa",             (14.9, 15.8, 73.7, 74.4), 162, 2.5),
    ("Maharashtra",     (16.0, 20.0, 73.0, 80.0), 168, 2.5),
    ("Odisha",          (18.0, 22.0, 82.0, 87.0), 172, 2.5),
    ("West Bengal",     (21.0, 24.5, 85.0, 89.5), 175, 2.5),
    ("Jharkhand",       (21.5, 25.0, 83.0, 87.5), 178, 2.5),
    ("Bihar",           (24.0, 27.5, 83.5, 88.5), 182, 2.5),
    ("Madhya Pradesh",  (21.0, 26.5, 74.0, 82.0), 183, 2.5),
    ("Uttar Pradesh",   (24.0, 30.5, 77.0, 84.5), 185, 2.5),
    ("Rajasthan",       (23.0, 30.0, 69.5, 77.5), 192, 2.5),
    ("Delhi NCR",       (28.0, 29.5, 76.5, 77.5), 190, 2.5),
    ("Punjab & Haryana",(28.5, 32.5, 73.5, 77.5), 195, 2.5),
    ("Gujarat",         (20.0, 24.5, 68.0, 74.5), 178, 2.5),
    ("Telangana",       (15.8, 19.5, 77.0, 81.5), 170, 2.5),
    ("Andhra Pradesh",  (12.6, 19.0, 76.5, 84.7), 166, 2.5),
    ("Tamil Nadu",      ( 8.0, 13.5, 76.9, 80.5), 160, 2.5),
    ("Northeast India", (22.0, 29.5, 89.5, 97.5), 148, 2.5),
    ("Himachal Pradesh",(30.0, 33.5, 75.5, 79.0), 198, 2.5),
    ("Uttarakhand",     (28.5, 31.5, 77.5, 81.0), 196, 2.5),
]

def detect_monsoon_onset(rain_da, target_year=None):
    import pandas as pd
    times = pd.to_datetime(rain_da.time.values)
    available_years = sorted(list(set(times.year)))
    
    if target_year is None or target_year not in available_years:
        target_year = available_years[-1]
        
    # Always slice rain_da to the target_year
    year_indices = np.where(times.year == target_year)[0]
    rain_da = rain_da.isel(time=year_indices)
    times = pd.to_datetime(rain_da.time.values)

    lats  = rain_da.lat.values
    lons  = rain_da.lon.values
    results = []
    for (name, (lat_min, lat_max, lon_min, lon_max), normal_doy, threshold) in MONSOON_WAYPOINTS:
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        if lat_mask.sum() == 0 or lon_mask.sum() == 0:
            results.append({"name": name, "status": "No Data", "onset_date": None,
                            "normal_doy": normal_doy, "days_vs_normal": None, "actual_year": target_year})
            continue
        region_rain = rain_da.values[:, lat_mask, :][:, :, lon_mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            daily_mean = np.nanmean(region_rain, axis=(1, 2))
        onset_idx = None
        streak = 0
        for i, (t, val) in enumerate(zip(times, daily_mean)):
            if t.dayofyear < normal_doy - 20:
                streak = 0
                continue
            if not np.isnan(val) and val >= threshold:
                streak += 1
                if streak >= 5:
                    onset_idx = i - 4
                    break
            else:
                streak = 0
        if onset_idx is not None:
            onset_date = times[onset_idx]
            days_diff = int(onset_date.dayofyear - normal_doy)
            results.append({"name": name, "status": "Arrived", "onset_date": onset_date.strftime("%d %b %Y"),
                            "normal_doy": normal_doy, "days_vs_normal": days_diff, "actual_year": target_year})
        else:
            cur_doy = times[-1].dayofyear
            status = "Delayed" if cur_doy > normal_doy + 7 else ("Awaited" if cur_doy > normal_doy - 20 else "Upcoming")
            results.append({"name": name, "status": status, "onset_date": None,
                            "normal_doy": normal_doy, "days_vs_normal": None, "actual_year": target_year})
    return results

def compute_exceedance_probability(predictions, lower_bounds, upper_bounds, threshold_mm):
    """P(rainfall > threshold) using Gaussian approx from MC Dropout. Returns (lat, lon) grid."""
    try:
        from scipy.stats import norm
    except ImportError:
        return np.full(predictions.shape[1:], np.nan)
    std_approx = np.maximum((upper_bounds - lower_bounds) / 2.0, 0.01)
    prob_per_day = 1.0 - norm.cdf(threshold_mm, loc=predictions, scale=std_approx)
    prob_not_exceed = np.where(np.isnan(prob_per_day), 1.0, 1.0 - prob_per_day)
    prob_any_day = 1.0 - np.prod(prob_not_exceed, axis=0)
    nan_mask = np.isnan(predictions[0])
    prob_any_day[nan_mask] = np.nan
    return prob_any_day

def compute_brier_skill_score(obs_values, prob_forecasts, threshold):
    valid = ~(np.isnan(obs_values) | np.isnan(prob_forecasts))
    if valid.sum() < 10:
        return {"brier_score": np.nan, "bss": np.nan, "clim_prob": np.nan, "n_events": 0, "n_total": 0}
    obs   = obs_values[valid]
    probs = prob_forecasts[valid]
    obs_bin   = (obs >= threshold).astype(float)
    bs        = float(np.mean((probs - obs_bin) ** 2))
    clim_prob = float(np.mean(obs_bin))
    bs_clim   = float(np.mean((clim_prob - obs_bin) ** 2))
    bss       = float(1.0 - bs / bs_clim) if bs_clim > 0 else np.nan
    return {"brier_score": bs, "bss": bss, "clim_prob": clim_prob,
            "n_events": int(obs_bin.sum()), "n_total": int(valid.sum())}

def compute_reliability_diagram_data(obs_values, prob_forecasts, threshold, n_bins=10):
    valid = ~(np.isnan(obs_values) | np.isnan(prob_forecasts))
    if valid.sum() < 20:
        return None
    obs      = obs_values[valid]
    probs    = prob_forecasts[valid]
    obs_bin  = (obs >= threshold).astype(float)
    edges    = np.linspace(0, 1, n_bins + 1)
    centers  = (edges[:-1] + edges[1:]) / 2
    obs_freq = []
    counts   = []
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i+1])
        obs_freq.append(float(np.mean(obs_bin[mask])) if mask.sum() > 0 else np.nan)
        counts.append(int(mask.sum()))
    return {"bin_centers": centers.tolist(), "obs_freq": obs_freq, "bin_counts": counts}
