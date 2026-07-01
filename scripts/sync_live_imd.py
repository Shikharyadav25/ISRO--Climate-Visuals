import os
import datetime
import xarray as xr
import imdlib as imd
import numpy as np
import time

def sync_live_data():
    data_dir = 'data/processed'
    datasets = [
        ('rain', 'IMD_Gridded_Rainfall_0.25_Real_v4.nc', 'rainfall', 'rain'),
        ('tmax', 'IMD_Gridded_MaxTemp_1.0_Real_v3.nc', 'max_temp', 'tmax'),
        ('tmin', 'IMD_Gridded_MinTemp_1.0_Real_v3.nc', 'min_temp', 'tmin')
    ]
    
    target_date = datetime.date.today() - datetime.timedelta(days=2) # 2 days lag is safer for IMD
    any_updates = False
    
    for var_type, file_name, var_name, imd_var_name in datasets:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            continue
            
        print(f"\nChecking {file_name} for missing days...")
        ds_old = xr.open_dataset(file_path, engine='netcdf4')
        last_date_np = ds_old.time.values[-1]
        
        import pandas as pd
        last_date = pd.to_datetime(last_date_np).date()
        
        if last_date >= target_date:
            print(f"{var_type} is already up to date ({last_date}).")
            ds_old.close()
            continue
            
        while last_date < target_date:
            start_date = last_date + datetime.timedelta(days=1)
            
            # Batch into max 30 days to avoid IMD server connection resets
            end_date = min(target_date, start_date + datetime.timedelta(days=30))
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching {var_type} from {start_str} to {end_str}...")
            try:
                data = imd.get_real_data(var_type, start_str, end_str, file_dir=os.path.join(data_dir, 'temp_imdlib'))
                ds_new = data.get_xarray()
                time.sleep(1) # Be nice to IMD servers
            except Exception as e:
                print(f"Failed to fetch {var_type}: {e}")
                break # Exit the while loop and move to next variable
                
            ds_new = ds_new.rename({imd_var_name: var_name})
            
            # Clean IMD Missing Values (-999.0 for rain, 99.9 for temp)
            if var_type == 'rain':
                ds_new[var_name] = xr.where(ds_new[var_name] <= -990.0, np.nan, ds_new[var_name])
            else:
                ds_new[var_name] = xr.where(ds_new[var_name] >= 99.0, np.nan, ds_new[var_name])
                
            if 'latitude' in ds_new.dims: ds_new = ds_new.rename({'latitude': 'lat'})
            if 'longitude' in ds_new.dims: ds_new = ds_new.rename({'longitude': 'lon'})
            
            if 'lat' in ds_new.dims and 'lat' in ds_old.dims:
                if len(ds_new.lat) != len(ds_old.lat):
                    ds_new = ds_new.interp(lat=ds_old.lat, lon=ds_old.lon, method='nearest')
                else:
                    ds_new['lat'] = ds_old['lat']
                    ds_new['lon'] = ds_old['lon']
                    
            ds_combined = xr.concat([ds_old, ds_new], dim='time')
            ds_combined[var_name] = ds_combined[var_name].astype(np.float32)
            
            temp_path = os.path.join(data_dir, f"temp_sync_{file_name}")
            ds_combined.to_netcdf(temp_path, engine='netcdf4')
            
            ds_old.close()
            ds_combined.close()
            
            os.replace(temp_path, file_path)
            print(f"Successfully updated {file_name} to {end_str}!")
            any_updates = True
            
            # Reopen the newly updated dataset for the next iteration of the chunk loop
            ds_old = xr.open_dataset(file_path, engine='netcdf4')
            last_date = end_date
            
        ds_old.close()
        
    return any_updates

if __name__ == '__main__':
    sync_live_data()
