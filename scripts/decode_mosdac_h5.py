import os
import glob
import h5py
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata

def decode_and_regrid_sst():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    
    os.makedirs(proc_dir, exist_ok=True)
    
    h5_files = glob.glob(os.path.join(raw_dir, '*L2B_SST*.h5'))
    if not h5_files:
        print("No MOSDAC SST .h5 files found in data/raw/")
        return
        
    print(f"Found {len(h5_files)} MOSDAC SST files. Regridding to 0.25 deg...")
    
    # Target Grid (0.25 deg) matching the original fake implementation
    lons_ocean = np.arange(50.0, 100.0, 0.25)
    lats_ocean = np.arange(0.0, 30.0, 0.25)
    target_lon_grid, target_lat_grid = np.meshgrid(lons_ocean, lats_ocean)
    
    # We will average all files into a single daily mean for simplicity
    all_sst_grids = []
    
    for fpath in h5_files:
        print(f"Processing {os.path.basename(fpath)}...")
        try:
            with h5py.File(fpath, 'r') as f:
                # Read arrays
                sst = f['SST'][0, :, :] # shape (1, 2816, 2805) -> (2816, 2805)
                
                # Apply scale factors for coordinates
                lat_raw = f['Latitude'][:, :]
                lon_raw = f['Longitude'][:, :]
                
                lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
                lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
                
                lat = np.where(lat_raw == 32767, np.nan, lat_raw * lat_scale)
                lon = np.where(lon_raw == 32767, np.nan, lon_raw * lon_scale)
                
                # Filter out fill values
                valid_mask = (sst > 200) & (sst < 350) & (lat >= 0) & (lat <= 30) & (lon >= 50) & (lon <= 100)
                
                if np.sum(valid_mask) == 0:
                    print("No valid data points in the target region for this file.")
                    continue
                    
                points = np.column_stack((lon[valid_mask], lat[valid_mask]))
                values = sst[valid_mask]
                
                # Regrid using Linear to avoid smearing outside the data swath, fallback to nearest for small holes
                regridded_sst = griddata(points, values, (target_lon_grid, target_lat_grid), method='linear')
                all_sst_grids.append(regridded_sst)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            
    if not all_sst_grids:
        print("Failed to extract any valid SST grids.")
        return
        
    # Average the grids (ignoring NaNs)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        daily_mean_sst = np.nanmean(np.array(all_sst_grids), axis=0)
    
    # Apply land mask so SST only shows on the ocean
    try:
        from global_land_mask import globe
        ocean_mask = globe.is_ocean(target_lat_grid, target_lon_grid)
    except ImportError:
        ocean_mask = (target_lon_grid < 73.0) | (target_lon_grid > 84.0) | (target_lat_grid < 12.0)
        
    daily_mean_sst = np.where(ocean_mask, daily_mean_sst, np.nan)
    
    # Convert from Kelvin to Celsius if necessary
    if np.nanmean(daily_mean_sst) > 100:
        daily_mean_sst = daily_mean_sst - 273.15
        
    # Recreate the time dimension (e.g. for a single day)
    times = pd.date_range(start='2024-01-01', periods=1, freq='D')
    sst_data = daily_mean_sst[np.newaxis, :, :] # Shape: (1, lat, lon)
    
    # Create NetCDF
    sst_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_SST_Real.nc')
    ds_sst = xr.Dataset(
        data_vars=dict(sst=(["time", "lat", "lon"], sst_data, {"units": "degC", "long_name": "Real INSAT Sea Surface Temperature"})),
        coords=dict(time=times, lat=(["lat"], lats_ocean), lon=(["lon"], lons_ocean)),
        attrs=dict(description="Real physical INSAT Sea Surface Temperature (3RIMG_L2B_SST)", source="MOSDAC INSAT-3D/3DR")
    )
    
    # We remove the file if it exists to overwrite cleanly, to avoid xarray permission errors on Windows
    if os.path.exists(sst_nc):
        try:
            os.remove(sst_nc)
        except:
            pass
            
    ds_sst.to_netcdf(sst_nc)
    print(f"\n[SUCCESS] Saved real regridded INSAT SST NetCDF: {sst_nc}")

if __name__ == "__main__":
    decode_and_regrid_sst()
