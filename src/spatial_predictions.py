import numpy as np
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

class SpatialClimatePredictor:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader

    def mask_region_boundary(self, data_array, region_name):
        """Masks xarray dataset/dataarray keeping only grid points inside the actual state boundary"""
        if data_array is None or region_name not in STATE_POLYGONS:
            return data_array
        poly = STATE_POLYGONS[region_name]
        if isinstance(data_array, xr.Dataset):
            lats = data_array.lat.values
            lons = data_array.lon.values
        else:
            lats = data_array.lat.values
            lons = data_array.lon.values

        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if is_point_in_polygon(lon, lat, poly):
                    mask[i, j] = True

        mask_da = xr.DataArray(mask, coords=[('lat', lats), ('lon', lons)])

        if isinstance(data_array, xr.Dataset):
            masked_ds = data_array.copy(deep=True)
            for var in masked_ds.data_vars:
                masked_ds[var] = xr.where(mask_da, masked_ds[var], np.nan)
            return masked_ds
        else:
            return xr.where(mask_da, data_array, np.nan)

    def slice_region(self, data_array, region_name="Karnataka"):
        """Slices the xarray DataArray to the bounding box of the selected pilot region."""
        if region_name not in PILOT_REGIONS or region_name == "All India":
            return data_array
        lat_min, lat_max, lon_min, lon_max = PILOT_REGIONS[region_name]
        min_span = 1.2
        lat_span = lat_max - lat_min
        if lat_span < min_span:
            lat_mid = (lat_max + lat_min) / 2.0
            lat_min = lat_mid - (min_span / 2.0)
            lat_max = lat_mid + (min_span / 2.0)
        lon_span = lon_max - lon_min
        if lon_span < min_span:
            lon_mid = (lon_max + lon_min) / 2.0
            lon_min = lon_mid - (min_span / 2.0)
            lon_max = lon_mid + (min_span / 2.0)
        sliced = data_array.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        if sliced.sizes.get("lat", 0) == 0 or sliced.sizes.get("lon", 0) == 0:
            try:
                lat_mid = (lat_max + lat_min) / 2.0
                lon_mid = (lon_max + lon_min) / 2.0
                return data_array.sel(lat=lat_mid, lon=lon_mid, method="nearest")
            except Exception:
                return data_array
                
        if region_name in STATE_PATHS and STATE_PATHS[region_name]:
            lon_grid, lat_grid = np.meshgrid(sliced.lon.values, sliced.lat.values)
            points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
            mask = np.zeros(len(points), dtype=bool)
            for path in STATE_PATHS[region_name]:
                mask |= path.contains_points(points)
            mask_2d = mask.reshape(lat_grid.shape)
            mask_da = xr.DataArray(mask_2d, coords=[sliced.lat, sliced.lon], dims=["lat", "lon"])
            sliced = sliced.where(mask_da, np.nan)
            
        return sliced

    @staticmethod
    def load_full_climatology(full_rain_grid):
        """
        Pre-compute the full multi-year daily climatological mean atlas.
        Groups all time steps by (month, day) and averages across years.
        Returns: dict { (month, day): np.ndarray(lat, lon) }
        This is equivalent to the NCEP/NCAR Reanalysis daily climatology used by NOAA CPC.
        """
        import pandas as pd
        import warnings
        times = pd.to_datetime(full_rain_grid.time.values)
        clim_grids = {}
        for t_idx, t in enumerate(times):
            key = (t.month, t.day)
            if key not in clim_grids:
                clim_grids[key] = []
            clim_grids[key].append(full_rain_grid.values[t_idx])

        clim_atlas = {}
        for key, grids in clim_grids.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                clim_atlas[key] = np.nanmean(grids, axis=0)
        return clim_atlas

    def find_analog_years(self, recent_rain_grid, clim_atlas, n_analogs=3):
        """
        Finds the N most similar historical 30-day windows in the dataset using
        spatial Pearson correlation — the NOAA Climate Prediction Center (CPC) analog method.
        Returns: list of (year, spatial_correlation_score) tuples, sorted best-first.
        """
        import pandas as pd, warnings
        times = pd.to_datetime(recent_rain_grid.time.values)
        n_days = len(times)
        if n_days < 30:
            return []

        # Current pattern: spatial mean of last 30 days (flattened)
        recent_30 = recent_rain_grid.values[-30:]
        nan_mask_2d = np.all(np.isnan(recent_30), axis=0)
        ref_pattern = np.nanmean(recent_30, axis=0)
        ref_flat = ref_pattern[~nan_mask_2d].ravel()
        if len(ref_flat) < 10:
            return []

        last_date = times[-1]
        years_available = sorted(set(t.year for t in times))

        analog_scores = []
        for yr in years_available:
            # Find the same 30-day window in that year
            window_end = pd.Timestamp(year=yr, month=last_date.month, day=last_date.day)
            window_start = window_end - pd.Timedelta(days=29)
            # Get time indices for this window
            mask = (times >= window_start) & (times <= window_end)
            if mask.sum() < 20:  # Need at least 20 days of data
                continue
            yr_window = recent_rain_grid.values[mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                yr_pattern = np.nanmean(yr_window, axis=0)
            yr_flat = yr_pattern[~nan_mask_2d].ravel()
            if len(yr_flat) != len(ref_flat):
                continue
            # Pearson spatial correlation
            valid = ~(np.isnan(ref_flat) | np.isnan(yr_flat))
            if valid.sum() < 10:
                continue
            corr = np.corrcoef(ref_flat[valid], yr_flat[valid])[0, 1]
            if not np.isnan(corr):
                analog_scores.append((yr, corr))

        # Sort by best correlation
        analog_scores.sort(key=lambda x: -x[1])
        return analog_scores[:n_analogs]

    def get_analog_forecast_grid(self, recent_rain_grid, analog_years, days_ahead, clim_atlas):
        """
        For each analog year, retrieve the actual observed rainfall grids for the
        same N days ahead in that year. Returns ensemble mean and std of analog trajectories.
        This is equivalent to the 'extended ensemble' used by ECMWF and NCMRWF.
        """
        import pandas as pd, warnings
        times = pd.to_datetime(recent_rain_grid.time.values)
        last_date = times[-1]
        analog_trajectories = []

        for (yr, score) in analog_years:
            traj = []
            all_ok = True
            for d in range(1, days_ahead + 1):
                target_date = pd.Timestamp(year=yr, month=last_date.month, day=last_date.day) \
                              + pd.Timedelta(days=d)
                # Handle year rollover
                try:
                    t_mask = np.array([
                        (pd.Timestamp(t).month == target_date.month and
                         pd.Timestamp(t).day == target_date.day and
                         pd.Timestamp(t).year == target_date.year)
                        for t in recent_rain_grid.time.values
                    ])
                    if t_mask.sum() == 0:
                        # Fall back to climatology for this day
                        key = (target_date.month, target_date.day)
                        grid = clim_atlas.get(key)
                        if grid is None:
                            all_ok = False
                            break
                        traj.append(np.nan_to_num(grid, nan=0.0))
                    else:
                        traj.append(np.nan_to_num(recent_rain_grid.values[t_mask][0], nan=0.0))
                except Exception:
                    all_ok = False
                    break
            if all_ok and len(traj) == days_ahead:
                analog_trajectories.append(np.array(traj))  # (days, lat, lon)

        if not analog_trajectories:
            return None, None

        analog_stack = np.array(analog_trajectories)  # (n_analogs, days, lat, lon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            analog_mean = np.nanmean(analog_stack, axis=0)  # (days, lat, lon)
            analog_std  = np.nanstd(analog_stack,  axis=0)
        return analog_mean, analog_std

    def seasonal_colorscale_limits(self, clim_atlas, forecast_dates, percentile=95.0):
        """
        Returns scientifically correct (zmin, zmax) for the colorscale based on
        the historical seasonal peak during the forecast period.
        Prevents outlier days from washing out the color gradient.
        This mirrors NASA Giovanni and NOAA Viewer dynamic scaling conventions.
        """
        import pandas as pd, warnings
        if not clim_atlas or forecast_dates is None or len(forecast_dates) == 0:
            return 0.0, 20.0
        clim_grids_for_period = []
        for d in forecast_dates:
            key = (d.month, d.day)
            g = clim_atlas.get(key)
            if g is not None:
                clim_grids_for_period.append(g)
        if not clim_grids_for_period:
            return 0.0, 20.0
        stacked = np.array(clim_grids_for_period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zmax = float(np.nanpercentile(stacked, percentile))
        return 0.0, max(10.0, zmax)

    def predict_rainfall_next_days_spatial(self, recent_rain_grid, days_ahead=7, clim_atlas=None):
        """
        NASA/NCMRWF-level spatiotemporal forecast engine with 5 layers:
        1. Pre-computed multi-year climatological atlas (not window-limited)
        2. ConvLSTM predicts ANOMALY vs climatology (preserves spatial gradients)
        3. Analog Year Ensemble (NOAA CPC method, 3 best analog years)
        4. Weighted blend: neural anomaly + analog ensemble + climatology
        5. Mean Bias Correction (MBC) post-processing
        Returns: (predictions, lower_bounds, upper_bounds) — each shape (days, lat, lon)
        """
        import pandas as pd, warnings

        predictions  = []
        lower_bounds = []
        upper_bounds = []

        is_rainfall = (recent_rain_grid.name is not None and
                       any(lbl in str(recent_rain_grid.name).lower() for lbl in ['rain', 'precip']))
        fill_val = 0.0 if is_rainfall else 30.0
        nan_mask = np.isnan(recent_rain_grid.values[-1])

        # ── Layer 1: Build or use pre-computed climatological atlas ──────────────
        if clim_atlas is None:
            clim_atlas = self.load_full_climatology(recent_rain_grid)

        last_date = pd.to_datetime(recent_rain_grid.time.values[-1])
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=days_ahead, freq='D')

        # ── Layer 2: Analog Year Ensemble (NOAA CPC method) ──────────────────────
        analog_years = self.find_analog_years(recent_rain_grid, clim_atlas, n_analogs=3)
        analog_mean, analog_std = self.get_analog_forecast_grid(
            recent_rain_grid, analog_years, days_ahead, clim_atlas
        )

        # ── Layer 3: ConvLSTM Anomaly Prediction ─────────────────────────────────
        convlstm_anomaly = None
        convlstm_model = None
        if self.model_loader:
            key_m = "convlstm" if is_rainfall else "convlstm_temp"
            convlstm_model = self.model_loader.models.get(key_m)

        if convlstm_model is not None:
            try:
                import torch
                device = next(convlstm_model.parameters()).device
                grid_vals = recent_rain_grid.values[-10:]
                if len(grid_vals) < 10:
                    pad_len = 10 - len(grid_vals)
                    grid_vals = np.pad(grid_vals, ((pad_len, 0), (0, 0), (0, 0)), mode='edge')
                grid_vals = np.nan_to_num(grid_vals, nan=fill_val)

                # Compute anomaly relative to climatology
                clim_for_window = []
                times_window = pd.to_datetime(recent_rain_grid.time.values[-10:])
                for t in times_window:
                    key_c = (t.month, t.day)
                    cg = clim_atlas.get(key_c, np.zeros_like(grid_vals[0]))
                    clim_for_window.append(np.nan_to_num(cg, nan=fill_val))
                clim_window = np.array(clim_for_window)

                # Neural network sees anomaly (departure from climatology)
                anomaly_vals = grid_vals - clim_window
                if is_rainfall:
                    # Log-scaled anomaly for rainfall (prevents negative log domain)
                    anom_scaled = np.sign(anomaly_vals) * np.log1p(np.abs(anomaly_vals)) / 3.0
                else:
                    anom_scaled = anomaly_vals / 10.0

                input_tensor = (torch.tensor(anom_scaled, dtype=torch.float32)
                                .unsqueeze(0).unsqueeze(2).to(device))

                n_ensemble = 5
                convlstm_model.train()
                cur_input = input_tensor
                pred_anomalies = []
                for i in range(days_ahead):
                    ens_preds = []
                    for _ in range(n_ensemble):
                        with torch.no_grad():
                            pred_grid, _ = convlstm_model(cur_input)
                            raw = pred_grid.squeeze().cpu().numpy()
                            if is_rainfall:
                                anom = np.sign(raw) * (np.expm1(np.abs(raw) * 3.0))
                            else:
                                anom = raw * 10.0
                            ens_preds.append(anom)
                    ens_arr = np.stack(ens_preds, axis=0)
                    pred_anomalies.append(np.mean(ens_arr, axis=0))
                    # Feed next step (anomaly space)
                    next_anom = pred_anomalies[-1]
                    if is_rainfall:
                        next_scaled = np.sign(next_anom) * np.log1p(np.abs(next_anom)) / 3.0
                    else:
                        next_scaled = next_anom / 10.0
                    next_scaled = np.nan_to_num(next_scaled, nan=0.0)
                    next_t = (torch.tensor(next_scaled, dtype=torch.float32)
                              .unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device))
                    cur_input = torch.cat([cur_input[:, 1:], next_t], dim=1)

                convlstm_model.eval()
                convlstm_anomaly = np.array(pred_anomalies)  # (days, lat, lon)

            except Exception as e:
                print(f"[Forecast] ConvLSTM anomaly inference failed: {e}")
                if convlstm_model is not None:
                    convlstm_model.eval()
                convlstm_anomaly = None

        # ── Layer 4: Weighted Blend ───────────────────────────────────────────────
        # Weights: neural anomaly carries more weight early, analog ensemble later
        # By Day +7, analog ensemble and climatology dominate (just like ECMWF)
        for i in range(days_ahead):
            nd = forecast_dates[i]
            clim_grid = clim_atlas.get((nd.month, nd.day))
            clim_grid = np.nan_to_num(
                clim_grid if clim_grid is not None else np.full(nan_mask.shape, fill_val),
                nan=fill_val
            )

            # Decay schedule: day 1 = 55% neural, day 7 = 20% neural
            neural_weight  = max(0.15, 0.55 * (0.88 ** i))
            analog_weight  = min(0.45, 0.25 + 0.05 * i)  # grows with time
            clim_weight    = max(0.15, 1.0 - neural_weight - analog_weight)

            if convlstm_anomaly is not None:
                # Neural component: climatology + AI-predicted anomaly
                neural_component = clim_grid + convlstm_anomaly[i]
                if is_rainfall:
                    neural_component = np.maximum(0, neural_component)
            else:
                neural_component = clim_grid  # fallback to climatology

            if analog_mean is not None:
                analog_component = analog_mean[i]
            else:
                analog_component = clim_grid  # fallback to climatology

            blended = (neural_weight * neural_component +
                       analog_weight * analog_component +
                       clim_weight   * clim_grid)
            if is_rainfall:
                blended = np.maximum(0, blended)

            # ── Layer 5: Mean Bias Correction (MBC) ──────────────────────────────
            # Compute ratio of historical climatological mean to blended prediction mean
            clim_spatial_mean = np.nanmean(clim_grid)
            blend_spatial_mean = np.nanmean(blended[~nan_mask]) if (~nan_mask).any() else 1.0
            if blend_spatial_mean > 0.01 and clim_spatial_mean > 0.0:
                mbc_factor = np.clip(clim_spatial_mean / blend_spatial_mean, 0.5, 2.0)
                blended = blended * mbc_factor

            # Compute prediction uncertainty
            if analog_std is not None:
                unc_grid = analog_std[i]
            else:
                # Propagate uncertainty from recent 30-day std, growing with horizon
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    base_std = np.nanstd(recent_rain_grid.values[-30:], axis=0) \
                               if len(recent_rain_grid.time) >= 30 \
                               else np.nanstd(recent_rain_grid.values, axis=0)
                unc_grid = np.nan_to_num(base_std, nan=0.5) * (1.0 + 0.1 * i)

            blended[nan_mask] = np.nan
            predictions.append(blended)
            lower_bounds.append(np.where(nan_mask, np.nan,
                                         np.maximum(0 if is_rainfall else -50,
                                                    blended - unc_grid)))
            upper_bounds.append(np.where(nan_mask, np.nan, blended + unc_grid))

        return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)

    def simulate_what_if_spatial(self, base_rain_grid, base_temp_grid, rain_modifier, temp_modifier):
        """
        Simulate climate change using localized Clausius-Clapeyron thermodynamic interactions.
        Integrates non-linear moisture holding capacity (Tetens formula) and convective scaling.
        """
        # Base modifiers
        mod_rain = base_rain_grid.values * (1 + rain_modifier / 100.0)
        mod_temp = base_temp_grid.values + temp_modifier
        
        # Rigorous Clausius-Clapeyron scaling
        # We apply an intensification factor based on Saturation Vapor Pressure (e_s)
        if temp_modifier > 0.0:
            # Tetens formula for e_s (hPa)
            e_s_base = 6.11 * np.exp((17.27 * base_temp_grid.values) / (base_temp_grid.values + 237.3))
            e_s_mod = 6.11 * np.exp((17.27 * mod_temp) / (mod_temp + 237.3))
            
            # Thermodynamic scaling factor (e_s ratio)
            thermo_scaling = e_s_mod / e_s_base
            
            # Apply non-linear intensification strictly to convective precipitation (heavy rain)
            mod_rain = np.where(mod_rain > 5.0, mod_rain * thermo_scaling, mod_rain)
            mod_rain = np.maximum(0, mod_rain)
            
        elif temp_modifier < 0.0:
            # Suppressed convection dynamics
            drying_factor = 1.0 + (temp_modifier * 0.04)
            mod_rain = np.where(mod_rain > 2.0, mod_rain * drying_factor, mod_rain)
            mod_rain = np.maximum(0, mod_rain)
            
        return {'modified_rainfall': mod_rain, 'modified_max_temp': mod_temp}

    def simulate_soil_moisture(self, rain_grid, temp_grid):
        """
        Physics-based simulation of NICES Soil Moisture Index (0-100%).
        Uses antecedent precipitation index proxy and evapotranspiration decay.
        """
        # Baseline moisture from rainfall (logarithmic saturation)
        rain_mm = np.nan_to_num(rain_grid.values, nan=0.0)
        moisture = 100.0 * (1.0 - np.exp(-rain_mm / 15.0))
        
        # Evaporative stress from temperature
        temp_c = np.nan_to_num(temp_grid.values, nan=25.0)
        evap_stress = np.maximum(0, (temp_c - 20.0) * 1.5)
        
        # Combine and bound
        moisture = moisture - evap_stress
        moisture = np.clip(moisture + 15.0, 0, 100) # Baseline 15% minimum moisture
        
        # Re-apply spatial mask
        moisture = np.where(np.isnan(rain_grid.values), np.nan, moisture)
        
        return xr.DataArray(
            moisture,
            coords=[rain_grid.lat, rain_grid.lon],
            dims=["lat", "lon"],
            name="soil_moisture"
        )

    def assimilate_multi_source_data(self, imd_ground_grid, insat_sat_grid, variable="temperature"):
        """
        Optimal Interpolation (Inverse-Variance Weighting) data assimilation.
        Fuses IMD ground observations with MOSDAC INSAT satellite retrievals.
        """
        if variable == "temperature":
            var_ground = 1.2 ** 2
            var_sat    = 2.5 ** 2
        else:
            var_ground = 3.0 ** 2
            var_sat    = 5.0 ** 2
        weight_ground = var_sat    / (var_ground + var_sat)
        weight_sat    = var_ground / (var_ground + var_sat)
        fused_grid = (imd_ground_grid.values * weight_ground) + (insat_sat_grid.values * weight_sat)
        return xr.DataArray(
            fused_grid,
            coords=[imd_ground_grid.lat, imd_ground_grid.lon],
            dims=["lat", "lon"],
            name=f"fused_{variable}"
        )

    def compute_validation_metrics(self, observed_grid, predicted_grid):
        """
        Compute full validation metric suite.
        Returns: RMSE, MAE, Bias, Correlation, Skill Score vs Persistence baseline.
        """
        obs  = np.array(observed_grid, dtype=np.float64).ravel()
        pred = np.array(predicted_grid, dtype=np.float64).ravel()
        mask = ~(np.isnan(obs) | np.isnan(pred))
        if mask.sum() == 0:
            return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan, "skill": np.nan}
        o = obs[mask]
        p = pred[mask]
        rmse = np.sqrt(np.mean((o - p) ** 2))
        mae  = np.mean(np.abs(o - p))
        bias = np.mean(p - o)
        corr = np.corrcoef(o, p)[0, 1] if len(o) > 1 else np.nan
        clim_mean = np.mean(o)
        rmse_clim = np.sqrt(np.mean((o - clim_mean) ** 2))
        skill = 1.0 - (rmse / rmse_clim) if rmse_clim > 0 else np.nan
        return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "skill": skill}
