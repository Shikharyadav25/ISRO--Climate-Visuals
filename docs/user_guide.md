# Streamlit GIS Console Operation and User Guide

This document provides step-by-step instructions for operating the interactive digital twin dashboard (`app/streamlit_app.py`).

---

## Navigation and Console Tabs

The dashboard contains six core functional tabs accessible from the top navigation menu:

### 1. 2D Spatial Reanalysis Console
* **Variables Displayed:** Ground observations (Precipitation, Maximum Temperature, Minimum Temperature), satellite telemetry (INSAT-3D/3DR Land Surface Temperature, Sea Surface Temperature), and NICES Soil Moisture.
* **Geospatial Projections:** Renders the grids using Plotly Express `Scattermap` maps overlaid on Mapbox dark styles.
* **Boundary Clipping Selection:** Use the sidebar dropdown menu to select a pilot region (e.g. Karnataka, Uttar Pradesh, Odisha) or the entire All-India domain. Grid points outside the selected vector boundary will be automatically masked to `NaN`.

### 2. 7-Day Forecasting Engine and 4D Playback
* **Playback Controls:** Use the slider or click the "Play" button at the bottom of the map to execute a temporal animation of the 7-day predicted grids.
* **Uncertainty Corridor:** Standard deviation bounds (±1σ) generated via Monte Carlo Dropout are toggled on the map to visualize grid cell forecast confidence.
* **Probabilistic Exceedance Map:** Displays the probability (0% to 100%) that the target parameter (e.g., rainfall) will exceed a user-defined threshold.

### 3. Climate Indices and Hydrological Hazards
* **WMO Drought Tracker:** Maps the 30-day (SPI-30) and 90-day (SPI-90) Standardized Precipitation Index to locate soil moisture deficits.
* **FAO-56 Crop Water Stress:** Displays evapotranspiration deficit indices (0.0 to 1.0) to highlight irrigation requirements.
* **Flash Flood Guidance (FFG):** Warns of potential flooding by calculating saturated soil capacity.
* **Monsoon Onset Tracker:** A table of 20 regional waypoints with metrics cards detailing the arrival dates and arrival delay deltas (e.g., `Arrived · -19d early`).

### 4. What-If Scenario Simulator
* **Precipitation Slider:** Scale rainfall grids from -100% (extreme drought) to +100% (flooding).
* **Temperature Slider:** Apply constant offsets from -5°C to +5°C.
* **Dual Map Comparison:** Shows the baseline observed grid side-by-side with the simulated what-if output, along with a zero-centered diverging anomaly grid (Simulated - Baseline).
* **CSV Export:** Click the "Export Grid Data to CSV" button to download coordinates and simulated values.

### 5. Model Verification and Calibration
* **Calibration Curves:** Plots forecast probabilities against observed frequencies in 10 bins.
* **Holdout Diagnostics:** Evaluates the forecasting engine on the 2022-2023 holdout period, displaying the Brier Score, Brier Skill Score, RMSE, and MAE.

### 6. Conversational Crop Advisory Copilot
* **Interaction:** Type natural language agricultural questions into the chat bar.
* **Grounding:** The copilot retrieves district contingency guidelines matching the currently active region, combines them with the forecasted weather anomalies, and outputs agricultural advice.

---

## Legend and Metric Interpretation

* **Rainfall anomalies:** Color scale uses a green-to-blue projection. Blue indicates heavy rain, green represents light rain, and white/transparent denotes dry grid cells.
* **Temperature anomalies:** Uses a blue-yellow-red diverging palette. Red marks positive temperature anomalies (heatwaves), blue marks negative anomalies (coldwaves), and yellow represents seasonal averages.
* **What-If Anomaly Maps:** Uses a zero-centered diverging colorscale (e.g., red-to-blue or brown-to-green) to isolate absolute differences.
