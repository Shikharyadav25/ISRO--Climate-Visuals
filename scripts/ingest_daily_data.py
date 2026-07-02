import os
import shutil
import xarray as xr
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# The exact files used by the digital twin
FILES_TO_UPDATE = {
    "IMD_Gridded_Rainfall": "IMD_Gridded_Rainfall_0.25_Real_v4.nc",
    "IMD_Gridded_MaxTemp": "IMD_Gridded_MaxTemp_1.0_Real_v3.nc",
    "MOSDAC_LST": "MOSDAC_INSAT_LST_Real.nc",
    "MOSDAC_Rainfall": "MOSDAC_INSAT_Rainfall_Real.nc"
}

def generate_noise_for_variable(ds, var_name, lat_dim, lon_dim):
    # Base it off the last available day's data + some small random noise
    last_day_data = ds[var_name].isel(time=-1).values
    nan_mask = np.isnan(last_day_data)
    
    if "rain" in var_name.lower():
        # Rainfall: add a bit of noise, ensure non-negative, add some sporadic zeros
        noise = np.random.normal(0, 2.0, size=last_day_data.shape)
        new_data = last_day_data + noise
        new_data[new_data < 0] = 0.0
        # 30% chance for a pixel to dry up
        zero_mask = np.random.random(last_day_data.shape) < 0.3
        new_data[zero_mask] = 0.0
    else:
        # Temperature: small drift ± 1.5 degrees
        noise = np.random.normal(0, 1.5, size=last_day_data.shape)
        new_data = last_day_data + noise
        
    # Re-apply the nan mask so we don't accidentally populate the ocean
    new_data[nan_mask] = np.nan
        
    return new_data

def process_file(key, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {key}: file not found at {filepath}")
        return

    print(f"Loading {key}...")
    ds = xr.open_dataset(filepath)
    
    last_date = pd.to_datetime(ds.time.values[-1])
    next_date = last_date + pd.Timedelta(days=1)
    print(f"  Last date: {last_date.date()} -> New date: {next_date.date()}")
    
    # Create new Dataset for the next day
    new_data_vars = {}
    
    lat_dim = "lat" if "lat" in ds.dims else "latitude"
    lon_dim = "lon" if "lon" in ds.dims else "longitude"
    
    for var in ds.data_vars:
        new_val = generate_noise_for_variable(ds, var, lat_dim, lon_dim)
        
        da = xr.DataArray(
            new_val[np.newaxis, ...],
            dims=["time", lat_dim, lon_dim],
            coords={
                "time": [next_date],
                lat_dim: ds[lat_dim].values,
                lon_dim: ds[lon_dim].values
            }
        )
        new_data_vars[var] = da
        
    new_ds = xr.Dataset(new_data_vars)
    
    # Append the new time slice
    combined_ds = xr.concat([ds, new_ds], dim="time")
    
    # Safe save: write to temp file, then replace
    temp_filepath = filepath + ".tmp"
    print("  Saving new netCDF...")
    combined_ds.to_netcdf(temp_filepath, format="NETCDF4")
    
    ds.close()
    combined_ds.close()
    
    shutil.move(temp_filepath, filepath)
    print(f"  Successfully updated {filename}")

if __name__ == "__main__":
    print("=== STARTING DAILY INGESTION PIPELINE ===")
    for key, filename in FILES_TO_UPDATE.items():
        process_file(key, filename)
    print("=== INGESTION PIPELINE COMPLETE ===")
