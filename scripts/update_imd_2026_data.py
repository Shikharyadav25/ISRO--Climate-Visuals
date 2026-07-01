import os
import xarray as xr
import imdlib as imd
import numpy as np

def update_dataset(var_type, old_file_name, new_file_name, var_name_in_old, new_var_name, start_year, end_year):
    print(f"\n--- Processing {var_type.upper()} ({start_year}-{end_year}) ---")
    data_dir = 'data/processed'
    old_file_path = os.path.join(data_dir, old_file_name)
    new_file_path = os.path.join(data_dir, new_file_name)
    
    # 1. Download new data
    print(f"Downloading {var_type} data from IMD...")
    try:
        data = imd.get_data(var_type, start_year, end_year, fn_format='yearwise', file_dir=os.path.join(data_dir, 'temp_imdlib'))
        ds_new = data.get_xarray()
        print("Download successful!")
    except Exception as e:
        print(f"Failed to download {var_type} data: {e}")
        return

    # 2. Format new data to match the old dataset
    ds_new = ds_new.rename({new_var_name: var_name_in_old})
    if 'latitude' in ds_new.dims: ds_new = ds_new.rename({'latitude': 'lat'})
    if 'longitude' in ds_new.dims: ds_new = ds_new.rename({'longitude': 'lon'})

    # 3. Load old dataset
    print(f"Loading existing data from {old_file_name}...")
    ds_old = xr.open_dataset(old_file_path, engine='netcdf4')
    
    # 4. Filter dates
    last_old_date = ds_old.time.values[-1]
    ds_new = ds_new.sel(time=slice(last_old_date + np.timedelta64(1, 'D'), None))
    
    if len(ds_new.time) == 0:
        print("No new data to append. Skipping.")
        ds_old.close()
        return

    print(f"Appending {len(ds_new.time)} new days...")
    
    if 'lat' in ds_new.dims and 'lat' in ds_old.dims:
        if len(ds_new.lat) != len(ds_old.lat):
            print("Interpolating new data to match old grid resolution...")
            ds_new = ds_new.interp(lat=ds_old.lat, lon=ds_old.lon, method='nearest')
        else:
            ds_new['lat'] = ds_old['lat']
            ds_new['lon'] = ds_old['lon']

    # 5. Combine and save to NEW file path to avoid locking issues
    ds_combined = xr.concat([ds_old, ds_new], dim='time')
    ds_combined[var_name_in_old] = ds_combined[var_name_in_old].astype(np.float32)

    print(f"Saving combined dataset to {new_file_name}...")
    ds_combined.to_netcdf(new_file_path, engine='netcdf4')
    
    ds_old.close()
    ds_combined.close()
    print(f"Successfully created {new_file_name}!")

if __name__ == '__main__':
    # Max Temp
    update_dataset('tmax', 'IMD_Gridded_MaxTemp_1.0_Real.nc', 'IMD_Gridded_MaxTemp_1.0_Real_v2.nc', 'max_temp', 'tmax', 2024, 2025)
    # Min Temp
    update_dataset('tmin', 'IMD_Gridded_MinTemp_1.0_Real.nc', 'IMD_Gridded_MinTemp_1.0_Real_v2.nc', 'min_temp', 'tmin', 2024, 2025)
    
    print("\nAll datasets updated successfully!")
