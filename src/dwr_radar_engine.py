import urllib.request
import os
import time

class DopplerRadarEngine:
    def __init__(self):
        self.base_url = "https://mausam.imd.gov.in/Radar"
        # Bounding boxes are approximations (radar range ~250km from center)
        self.radars = {
            "Delhi": {"file": "MaxZ_Delhi.gif", "bbox": [74.5, 26.0, 79.5, 31.0]},
            "Mumbai": {"file": "MaxZ_Mumbai.gif", "bbox": [70.0, 16.5, 75.5, 21.5]},
            "Chennai": {"file": "MaxZ_Chennai.gif", "bbox": [77.5, 10.5, 82.5, 15.5]},
            "Kolkata": {"file": "MaxZ_Kolkata.gif", "bbox": [86.0, 20.0, 91.0, 25.0]},
            "Bhopal": {"file": "MaxZ_Bhopal.gif", "bbox": [74.5, 21.0, 79.5, 25.5]},
        }
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'radar')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_latest_radar(self, station="Delhi"):
        if station not in self.radars:
            return None
            
        file_name = self.radars[station]["file"]
        url = f"{self.base_url}/{file_name}"
        save_path = os.path.join(self.cache_dir, file_name)
        
        try:
            # We add a cache bust to ensure we get the live 10-min feed
            req = urllib.request.Request(f"{url}?t={int(time.time())}", headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                with open(save_path, 'wb') as out_file:
                    out_file.write(response.read())
            
            return {
                "station": station,
                "image_path": save_path,
                "bbox": self.radars[station]["bbox"],
                "status": "LIVE"
            }
        except Exception as e:
            print(f"Failed to fetch DWR for {station}: {e}")
            if os.path.exists(save_path):
                return {
                    "station": station,
                    "image_path": save_path,
                    "bbox": self.radars[station]["bbox"],
                    "status": "CACHED"
                }
            return None
