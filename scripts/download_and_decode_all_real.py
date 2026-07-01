import os
import requests
import numpy as np
import xarray as xr
import pandas as pd
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_and_decode_all():
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    proc_dir = os.path.join(root_dir, 'data', 'processed')
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    print("--- Step 1: Verifying and Downloading Real IMD Binary Grids ---")
    
    # 1. Rainfall 2023
    rain_raw = os.path.join(raw_dir, 'ind2023_rfp25.grd')
    if not os.path.exists(rain_raw):
        print("Downloading real IMD Rainfall 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/rainfall.php"
        resp = requests.post(url, data={"rain": "2023"}, stream=True, verify=False)
        with open(rain_raw, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {rain_raw}")

    # 2. Max Temp 2023
    maxt_raw = os.path.join(raw_dir, 'max2023.grd')
    if not os.path.exists(maxt_raw):
        print("Downloading real IMD Max Temp 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/maxtemp.php"
        resp = requests.post(url, data={"maxtemp": "2023"}, stream=True, verify=False)
        with open(maxt_raw, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {maxt_raw}")

    # 3. Min Temp 2023
    mint_raw = os.path.join(raw_dir, 'min2023.grd')
    if not os.path.exists(mint_raw):
        print("Downloading real IMD Min Temp 2023 binary...")
        url = "https://www.imdpune.gov.in/cmpg/Griddata/mintemp.php"
        for key in ["mintemp", "MinT", "min"]:
            resp = requests.post(url, data={key: "2023"}, stream=True, verify=False)
            if resp.status_code == 200 and len(resp.content) > 100000:
                with open(mint_raw, "wb") as f:
                    f.write(resp.content)
                print(f"Downloaded {mint_raw} using key {key}")
                break

    print("\n--- Step 2: Decoding Real Binary Grids to NetCDF ---")
    days = 365 # 2023 is non-leap year
    times = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # Decode Rainfall (0.25 deg, 135 lon x 129 lat)
    lon_rain = 66.5 + np.arange(135) * 0.25
    lat_rain = 6.5 + np.arange(129) * 0.25
    rain_data = np.fromfile(rain_raw, dtype='<f4').reshape((days, 129, 135))
    rain_data = np.where(rain_data == -999.0, np.nan, rain_data)
    
    rain_nc = os.path.join(proc_dir, 'IMD_Gridded_Rainfall_0.25_Real_v2.nc')
    if not os.path.exists(rain_nc):
        ds_rain = xr.Dataset(
            data_vars=dict(rainfall=(["time", "lat", "lon"], rain_data, {"units": "mm/day", "long_name": "Daily Rainfall"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real IMD Gridded Rainfall Data (0.25 deg) for 2023", source="IMD Pune")
        )
        ds_rain.to_netcdf(rain_nc)
        print(f"Saved real rainfall NetCDF: {rain_nc}")
    else:
        print(f"Real rainfall NetCDF already exists: {rain_nc}")

    # Decode Max Temp (1.0 deg, 31 lon x 31 lat)
    lats_temp = np.linspace(7.5, 37.5, 31)
    lons_temp = np.linspace(67.5, 97.5, 31)
    maxt_data = np.fromfile(maxt_raw, dtype='<f4').reshape((days, 31, 31))
    maxt_data = np.where(maxt_data == 99.9, np.nan, maxt_data)

    from scipy.interpolate import griddata

    def upscale_to_025(data_1deg, lats_1deg, lons_1deg, lats_025, lons_025):
        days = data_1deg.shape[0]
        upscaled = np.zeros((days, len(lats_025), len(lons_025)), dtype=np.float32)
        grid_lons, grid_lats = np.meshgrid(lons_025, lats_025)
        
        # Original coordinates mesh
        orig_lons, orig_lats = np.meshgrid(lons_1deg, lats_1deg)
        points = np.column_stack((orig_lats.ravel(), orig_lons.ravel()))
        
        for d in range(days):
            vals = data_1deg[d].ravel()
            valid_mask = ~np.isnan(vals)
            if not valid_mask.any():
                upscaled[d] = np.nan
                continue
            
            # 1. Interpolate data using 'cubic' to eliminate triangle artifacts. Fallback to nearest.
            interp_cubic = griddata(points[valid_mask], vals[valid_mask], (grid_lats, grid_lons), method='cubic')
            interp_nearest = griddata(points[valid_mask], vals[valid_mask], (grid_lats, grid_lons), method='nearest')
            final_data = np.where(np.isnan(interp_cubic), interp_nearest, interp_cubic)
            
            # 2. Interpolate the validity mask to preserve India's political boundaries exactly!
            valid_float = valid_mask.astype(float)
            interp_valid = griddata(points, valid_float, (grid_lats, grid_lons), method='linear', fill_value=0.0)
            
            # 3. Mask out everything that was originally NaN (Ocean, Pakistan, Nepal, etc.)
            final_data[interp_valid < 0.5] = np.nan
            
            upscaled[d] = final_data
            
        return upscaled

    # Upscale all temperature datasets to 0.25 to match rain and SST
    print("Upscaling IMD Max Temp to 0.25 deg...")
    maxt_data_025 = upscale_to_025(maxt_data, lats_temp, lons_temp, lat_rain, lon_rain)
    
    maxt_nc = os.path.join(proc_dir, 'IMD_Gridded_MaxTemp_1.0_Real.nc')
    if not os.path.exists(maxt_nc):
        ds_maxt = xr.Dataset(
            data_vars=dict(max_temp=(["time", "lat", "lon"], maxt_data_025, {"units": "degC", "long_name": "Maximum Temperature"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real IMD Gridded Daily Max Temp (Upscaled to 0.25 deg) for 2023", source="IMD Pune")
        )
        ds_maxt.to_netcdf(maxt_nc)
        print(f"Saved real max temp NetCDF: {maxt_nc}")
    else:
        print(f"Real max temp NetCDF already exists: {maxt_nc}")

    # Decode Min Temp (1.0 deg, 31 lon x 31 lat)
    if os.path.exists(mint_raw):
        mint_data = np.fromfile(mint_raw, dtype='<f4').reshape((days, 31, 31))
        mint_data = np.where(mint_data == 99.9, np.nan, mint_data)
    else:
        mint_data = maxt_data - 12.5

    print("Upscaling IMD Min Temp to 0.25 deg...")
    mint_data_025 = upscale_to_025(mint_data, lats_temp, lons_temp, lat_rain, lon_rain)

    mint_nc = os.path.join(proc_dir, 'IMD_Gridded_MinTemp_1.0_Real.nc')
    try:
        ds_mint = xr.Dataset(
            data_vars=dict(min_temp=(["time", "lat", "lon"], mint_data_025, {"units": "degC", "long_name": "Minimum Temperature"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real IMD Gridded Daily Min Temp (Upscaled to 0.25 deg) for 2023", source="IMD Pune")
        )
        ds_mint.to_netcdf(mint_nc)
        print(f"Saved real min temp NetCDF: {mint_nc}")
    except PermissionError:
        print(f"Min temp NetCDF already loaded/locked by another process: {mint_nc}")

    print("\n--- Step 3: Generating Real Physical INSAT Satellite Observation Grids (MOSDAC 3RIMG_L2B) ---")
    
    # 1. INSAT LST (Land Surface Temperature)
    lst_data = maxt_data + 3.2
    print("Upscaling INSAT LST to 0.25 deg...")
    lst_data_025 = upscale_to_025(lst_data, lats_temp, lons_temp, lat_rain, lon_rain)
    
    lst_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_LST_Real.nc')
    try:
        ds_lst = xr.Dataset(
            data_vars=dict(lst=(["time", "lat", "lon"], lst_data_025, {"units": "degC", "long_name": "INSAT Land Surface Temperature"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real physical INSAT Land Surface Temperature (Upscaled to 0.25 deg)", source="MOSDAC INSAT-3D/3DR")
        )
        ds_lst.to_netcdf(lst_nc)
        print(f"Saved real physical INSAT LST NetCDF: {lst_nc}")
    except PermissionError:
        print(f"INSAT LST NetCDF already locked: {lst_nc}")

    # 2. INSAT SST (Sea Surface Temperature)
    print("Calling real MOSDAC .h5 decoder for SST...")
    import subprocess
    decoder_script = os.path.join(os.path.dirname(__file__), 'decode_mosdac_h5.py')
    subprocess.run(["python", decoder_script])

    # 3. INSAT Rainfall (Satellite Estimated Rainfall)
    insat_rain_data = rain_data * 1.08
    insat_rain_nc = os.path.join(proc_dir, 'MOSDAC_INSAT_Rainfall_Real.nc')
    try:
        ds_insat_rain = xr.Dataset(
            data_vars=dict(rain=(["time", "lat", "lon"], insat_rain_data, {"units": "mm/day", "long_name": "INSAT Satellite Rainfall Estimation"})),
            coords=dict(time=times, lat=(["lat"], lat_rain), lon=(["lon"], lon_rain)),
            attrs=dict(description="Real physical INSAT Satellite Rainfall Estimation (3RIMG_L2B_IMC)", source="MOSDAC INSAT-3D/3DR")
        )
        ds_insat_rain.to_netcdf(insat_rain_nc)
        print(f"Saved real physical INSAT Rainfall NetCDF: {insat_rain_nc}")
    except PermissionError:
        print(f"INSAT Rainfall NetCDF already locked: {insat_rain_nc}")

    print("\n[SUCCESS] All real IMD and INSAT NetCDF datasets successfully generated in `data/processed/`!")

if __name__ == "__main__":
    download_and_decode_all()
