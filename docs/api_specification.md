# FastAPI REST Web Service API Specification

This document details the REST endpoints, query parameters, JSON payload schemas, and client integrations for the API gateway microservice (`src/api/main.py`).

---

## Service Endpoints

### 1. Retrieve Ingestion Status
Returns metadata regarding the currently compiled NetCDF4 reanalysis files on disk.
* **Route:** `GET /api/status`
* **Query Parameters:** None
* **Success Response (200 OK):**
  ```json
  {
    "status": "operational",
    "rainfall_last_updated": "2023-12-31",
    "temperature_last_updated": "2023-12-31",
    "spatial_coverage": {
      "latitude": [6.5, 38.5],
      "longitude": [66.5, 100.0]
    }
  }
  ```

### 2. Fetch 7-Day Rainfall Forecast
Returns the 7-day spatial precipitation grid predictions for a specified administrative state.
* **Route:** `GET /api/predictions/rainfall`
* **Query Parameters:**
  * `state` (string, required): The target administrative state name (e.g. `Karnataka`, `Uttar Pradesh`, `Odisha`).
  * `target_date` (string, optional): ISO date format `YYYY-MM-DD`. Defaults to the latest available reanalysis date.
* **Success Response (200 OK):**
  ```json
  {
    "state": "Karnataka",
    "target_date": "2023-06-01",
    "forecast_steps": [1, 2, 3, 4, 5, 6, 7],
    "coordinates": {
      "latitude": [11.5, 11.75, 12.0],
      "longitude": [74.0, 74.25, 74.5]
    },
    "grid_shape": [7, 3, 3],
    "predictions": [
      [[0.0, 1.2, 0.5], [0.1, 0.0, 2.3], [1.5, 3.2, 0.0]],
      [[0.0, 0.8, 0.2], [0.0, 0.0, 1.1], [0.9, 2.1, 0.0]]
    ]
  }
  ```

### 3. Fetch 7-Day Temperature Forecast
Returns the 7-day spatial maximum and minimum temperature forecasts.
* **Route:** `GET /api/predictions/temperature`
* **Query Parameters:**
  * `state` (string, required): The target state name.
  * `target_date` (string, optional): ISO date format `YYYY-MM-DD`.
* **Success Response (200 OK):**
  ```json
  {
    "state": "Karnataka",
    "target_date": "2023-06-01",
    "forecast_steps": [1, 2, 3, 4, 5, 6, 7],
    "predictions": {
      "max_temp": [
        [[32.5, 33.1], [31.9, 32.4]],
        [[33.0, 33.5], [32.2, 32.8]]
      ],
      "min_temp": [
        [[22.1, 22.5], [21.8, 22.0]],
        [[22.4, 22.8], [22.0, 22.3]]
      ]
    }
  }
  ```

---

## Client Integration Examples

### 1. Python Request Example
```python
import requests

url = "http://localhost:8000/api/predictions/rainfall"
params = {
    "state": "Karnataka",
    "target_date": "2023-06-01"
}

response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    predictions = data["predictions"]
    latitudes = data["coordinates"]["latitude"]
    print(f"Retrieved grid shape: {data['grid_shape']}")
else:
    print(f"Error querying API: {response.status_code}")
```

### 2. cURL Example
```bash
curl -G "http://localhost:8000/api/predictions/rainfall" \
  --data-urlencode "state=Karnataka" \
  --data-urlencode "target_date=2023-06-01"
```
