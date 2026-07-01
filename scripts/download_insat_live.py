import os
import json
import subprocess
import shutil

def download_insat_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mosdac_api_dir = os.path.join(project_root, 'scripts', 'mosdac_api')
    creds_file = os.path.join(project_root, 'config.json')
    api_config_file = os.path.join(mosdac_api_dir, 'config.json')
    download_dir = os.path.join(project_root, 'data', 'raw')
    
    os.makedirs(download_dir, exist_ok=True)
    
    # 1. Load the user's credentials
    with open(creds_file, 'r') as f:
        creds = json.load(f)
        
    # 2. Configure the MOSDAC mdapi config.json
    api_config = {
        "user_credentials": {
            "username/email": creds["mosdac_username"],
            "password": creds["mosdac_password"]
        },
        "search_parameters": {
            "datasetId": "3RIMG_L2B_SST", # Starting with SST
            "startTime": "2024-01-01",
            "endTime": "2024-01-02",
            "count": "5",
            "boundingBox": "",
            "gId": ""
        },
        "download_settings": {
            "download_path": download_dir + "/",
            "organize_by_date": False,
            "skip_user_input": True, # Very important so it doesn't hang!
            "generate_error_logs": True,
            "error_logs_dir": mosdac_api_dir + "/"
        }
    }
    
    with open(api_config_file, 'w') as f:
        json.dump(api_config, f, indent=4)
        
    print(f"Configured mdapi.py to download to {download_dir}")
    print("Running mdapi.py...")
    
    # 3. Run the official mdapi script
    result = subprocess.run(["python", "mdapi.py"], cwd=mosdac_api_dir, capture_output=True, text=True)
    
    print("--- Output ---")
    print(result.stdout)
    if result.stderr:
        print("--- Errors ---")
        print(result.stderr)
        
if __name__ == "__main__":
    download_insat_data()
