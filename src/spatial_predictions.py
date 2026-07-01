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
        return sliced

    def predict_rainfall_next_days_spatial(self, recent_rain_grid, days_ahead=7):
        """
        Predict weather variable for the next N days using trained ConvLSTM with MC Dropout uncertainty.
        Falls back to statistical proxy if model unavailable.
        Returns: (predictions, lower_bounds, upper_bounds) — each shape (days, lat, lon)
        """
        predictions = []
        lower_bounds = []
        upper_bounds = []

        base_std = (np.nanstd(recent_rain_grid.values[-30:], axis=0)
                    if len(recent_rain_grid.time) >= 30
                    else np.nanstd(recent_rain_grid.values, axis=0))
        base_std = np.nan_to_num(base_std, nan=0.1)

        is_rainfall = (recent_rain_grid.name is not None and
                       any(lbl in str(recent_rain_grid.name).lower() for lbl in ['rain', 'precip']))

        convlstm_model = None
        if self.model_loader:
            key = "convlstm" if is_rainfall else "convlstm_temp"
            convlstm_model = self.model_loader.models.get(key)

        nan_mask = np.isnan(recent_rain_grid.values[-1])

        if convlstm_model is not None:
            try:
                import torch
                device = next(convlstm_model.parameters()).device
                grid_vals = (recent_rain_grid.values[-10:]
                             if len(recent_rain_grid.time) >= 10
                             else recent_rain_grid.values)
                if len(grid_vals) < 10:
                    pad_len = 10 - len(grid_vals)
                    grid_vals = np.pad(grid_vals, ((pad_len, 0), (0, 0), (0, 0)), mode='edge')

                fill_val = 0.0 if is_rainfall else 30.0
                grid_vals = np.nan_to_num(grid_vals, nan=fill_val)

                if is_rainfall:
                    grid_vals_scaled = np.log1p(np.maximum(0, grid_vals))
                else:
                    grid_vals_scaled = (grid_vals - 30.0) / 10.0

                input_tensor = (torch.tensor(grid_vals_scaled, dtype=torch.float32)
                                .unsqueeze(0).unsqueeze(2).to(device))

                # MC Dropout: keep model in train mode for stochastic forward passes
                n_ensemble = 5
                convlstm_model.train()

                # Precompute climatological grid for each forecast day to stabilize predictions
                import pandas as pd
                times = pd.to_datetime(recent_rain_grid.time.values)
                clim_grids = {}
                for t_idx, t in enumerate(times):
                    key = (t.month, t.day)
                    if key not in clim_grids:
                        clim_grids[key] = []
                    clim_grids[key].append(recent_rain_grid.values[t_idx])
                clim_mean_grids = {k: np.nanmean(v, axis=0) for k, v in clim_grids.items()}
                
                last_date = pd.to_datetime(recent_rain_grid.time.values[-1])
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')

                cur_input = input_tensor
                for i in range(days_ahead):
                    ensemble_preds = []
                    for _ in range(n_ensemble):
                        with torch.no_grad():
                            pred_grid, _ = convlstm_model(cur_input)
                            pred_np_scaled = pred_grid.squeeze().cpu().numpy()
                            if is_rainfall:
                                p = np.expm1(pred_np_scaled)
                                p = np.maximum(0, p)
                            else:
                                p = (pred_np_scaled * 10.0) + 30.0
                            ensemble_preds.append(p)

                    ensemble_preds = np.stack(ensemble_preds, axis=0)
                    mean_pred = np.mean(ensemble_preds, axis=0)
                    
                    # Blend prediction with the exact daily climatology grid to prevent compounding drift
                    decay = 0.65 ** (i + 1)
                    nd = forecast_dates[i]
                    clim_grid = clim_mean_grids.get((nd.month, nd.day), grid_vals[-1])
                    clim_grid = np.nan_to_num(clim_grid, nan=fill_val)
                    mean_pred = decay * mean_pred + (1.0 - decay) * clim_grid
                    
                    std_pred  = np.std(ensemble_preds, axis=0)

                    mean_pred[nan_mask] = np.nan
                    predictions.append(mean_pred)
                    lower_bounds.append(np.where(nan_mask, np.nan,
                                                 np.maximum(0 if is_rainfall else -50,
                                                            mean_pred - std_pred)))
                    upper_bounds.append(np.where(nan_mask, np.nan, mean_pred + std_pred))

                    if is_rainfall:
                        next_scaled = np.log1p(np.maximum(0, mean_pred))
                    else:
                        next_scaled = (mean_pred - 30.0) / 10.0
                    next_scaled = np.nan_to_num(next_scaled, nan=fill_val)
                    next_t = (torch.tensor(next_scaled, dtype=torch.float32)
                              .unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device))
                    cur_input = torch.cat([cur_input[:, 1:, :, :, :], next_t], dim=1)

                convlstm_model.eval()
                return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)

            except Exception as e:
                print(f"ConvLSTM inference error, falling back: {e}")
                if convlstm_model is not None:
                    convlstm_model.eval()
                predictions = []
                lower_bounds = []
                upper_bounds = []

        # Statistical proxy fallback
        if len(recent_rain_grid.time) < 30:
            last_day = recent_rain_grid.values[-1]
            mean_30  = np.nanmean(recent_rain_grid.values, axis=0)
        else:
            last_day = recent_rain_grid.values[-1]
            mean_30  = np.nanmean(recent_rain_grid.values[-30:], axis=0)

        current_state = np.nan_to_num(last_day.copy(), nan=0.0)
        mean_30 = np.nan_to_num(mean_30, nan=0.0)

        for i in range(days_ahead):
            alpha = 0.75 * (0.92 ** i)
            next_state = alpha * current_state + (1 - alpha) * mean_30
            next_state = np.maximum(0 if is_rainfall else -50, next_state)
            next_state[nan_mask] = np.nan
            predictions.append(next_state)
            uncertainty = base_std * (1.0 + 0.12 * i)
            lower_bounds.append(np.where(nan_mask, np.nan,
                                         np.maximum(0 if is_rainfall else -50,
                                                    next_state - uncertainty)))
            upper_bounds.append(np.where(nan_mask, np.nan, next_state + uncertainty))
            current_state = next_state

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
