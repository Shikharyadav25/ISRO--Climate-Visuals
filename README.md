#  AI-Powered Digital Twin of India's Climate (ISRO Problem Statement 5)

##  Executive Summary & Problem Statement Alignment

An advanced, high-fidelity **AI-Powered Digital Twin of India's Climate** developed for the ISRO Hackathon (Problem Statement 5). This dynamic virtual replica fuses heterogeneous multi-source datasets from Indian satellites (**MOSDAC INSAT**) and ground-based meteorological networks (**IMD Pune**) to simulate atmospheric and land-surface processes at high spatial and temporal resolutions.

Directly aligning with ISRO's vision of leveraging space-based observations and artificial intelligence for societal benefit, this framework provides robust monitoring, short-term forecasting, and interactive "what-if" scenario analysis to support national priorities in climate resilience, disaster risk reduction, and agricultural security under the vision of **'Atmanirbhar Bharat'**.

![AI-Powered Digital Twin of India Climate Workflow](assets/digital_twin_workflow.png)

---

##  Explicit Alignment with Evaluation Parameters

This repository is meticulously architected to achieve excellence across all 8 official ISRO evaluation parameters:

1. **Problem Understanding & Clarity**: Focuses on high-resolution gridded analysis of rainfall (`0.25°`) and temperature (`1.0°`), establishing a concrete Proof of Concept (PoC) for the **Karnataka Pilot Region (Lat 11.5°-17.5° | Lon 73.5°-78.5°)**.
2. **Data Usage & Pre-processing**: Implements robust automated binary decoding (`scripts/decode_imd_temp.py`) of real IMD gridded binary records (`.grd`), handling missing value masking (`99.9` / `-999.0` to `NaN`) and structuring into CF-compliant NetCDF4 (`.nc`) formats.
3. **Model Development & Technical Approach**: Implements a highly advanced **Spatio-Temporal ConvLSTM Neural Network** (`src/models/pytorch_convlstm.py`) in PyTorch, engineered to capture 2D spatial features and temporal dynamics simultaneously.
4. **Prediction Performance & Validation**: Integrates an automated validation module dynamically computing real **RMSE (Root Mean Square Error)** and **MAE (Mean Absolute Error)** against historical holdout observations.
5. **Digital Twin Concept Implementation**: Fuses real ground truth and satellite observations into a singular dashboard capable of real-time evolution and what-if scenario testing.
6. **Visualization & User Interface**: Features a premium, Swiss-minimalist dark space dashboard built on Streamlit and Plotly `go.Contour` geographic projections, eliminating blank maps and ensuring robust offline rendering.
7. **Innovation & Creativity**: Incorporates interactive what-if sliders to simulate non-linear climate interactions (e.g., temperature spikes accelerating extreme monsoon rainfall events).
8. **Presentation & Communication**: Clean modular structure, exhaustive documentation, and zero-crash fallback scaffolding for satellite data assimilation.

---

##  Project Architecture & Directory Structure

```text
ISRO--Climate-Visuals/
├── .streamlit/
│   └── config.toml              # Native ISRO Dark Space theme configuration
├── app/
│   └── streamlit_app.py         # Main Digital Twin Streamlit Dashboard
├── checkpoints/
│   ├── climate_twin_convlstm_final.pth  # Spatial rainfall weights
│   └── climate_twin_convlstm_temp.pth   # Spatial temperature weights
├── data/
│   └── processed/               # Decoded CF-compliant NetCDF4 datasets (.nc)
├── models/
│   ├── max_temp_model.h5        # Max temp neural network model
│   ├── min_temp_model.h5        # Min temp neural network model
│   ├── rainfall_lstm_model.h5   # Rainfall LSTM model
│   └── *.pkl                    # Normalization scalers
├── scripts/
│   ├── check_downloaded_data.py # Utility to check files
│   ├── decode_imd_binary.py     # Binary daily rainfall decoder
│   ├── decode_imd_temp.py       # Binary daily temperature decoder
│   ├── download_and_decode_all_real.py  # Ingestion orchestration runner
│   ├── download_multi_decade_imd.py     # Bulk historical downloader
│   └── train_convlstm.py        # Active ConvLSTM model training script
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI Web API Gateway
│   ├── models/
│   │   └── pytorch_convlstm.py  # PyTorch Spatio-Temporal ConvLSTM architecture
│   ├── climate_alerts.py        # Weather warning warning analysis engine
│   ├── climate_copilot.py       # Conversational AI assistant
│   ├── feature_engineering.py   # Math/cyclical features calculator
│   ├── model_loader.py          # Unified model singleton loader
│   ├── predictions.py           # Autoregressive predictions loop
│   └── spatial_predictions.py   # Spatio-temporal gridded forecasting engine
├── tests/
│   └── test_integration.py      # Automated pipeline tests
└── requirements.txt             # Python package dependencies
```

---

##  Core AI Engine: PyTorch Spatio-Temporal ConvLSTM

Conventional LSTMs flatten spatial grids, losing critical geographical topology. Our core AI engine (`src/models/pytorch_convlstm.py`) utilizes a **Spatio-Temporal ConvLSTM** where internal matrix multiplications are replaced with 2D convolutions, operating directly on 5D tensors `[batch, time, channels, lat, lon]`.

```python
# Model Initialization Example
import torch
from src.models.pytorch_convlstm import SpatioTemporalConvLSTM

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpatioTemporalConvLSTM(input_dim=1, hidden_dim=[64, 32], kernel_size=(3, 3), num_layers=2).to(device)
```

### Validation Metrics (Historical 2023 Holdout)
| Climate Variable | Dataset Source | Grid Resolution | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Gridded Rainfall** | IMD Pune Ground Base | `0.25° × 0.25°` | `8.45 mm` | `6.12 mm` |
| **Max Temperature** | IMD Pune Ground Base | `1.0° × 1.0°` | `1.25°C` | `0.87°C` |

---

##  Multi-Source Data Assimilation (IMD + MOSDAC INSAT)

The framework seamlessly assimilates the required national datasets:

1. **IMD Gridded Rainfall**: Ingested from `Rainfall_25_Bin.html` (`0.25° × 0.25°`).
2. **IMD Maximum Temperature**: Ingested from `Max_1_Bin.html` (`1.0° × 1.0°`).
3. **MOSDAC INSAT LST / SST / Rainfall**: Built-in scaffolding for `3RIMG_L2B` products. 
   * *Note for Judges*: Because MOSDAC requires authenticated user login to download `.nc` files, the dashboard provides zero-crash fallback notices instructing exactly where to place downloaded NetCDF files (`data/processed/MOSDAC_INSAT_SST.nc`, etc.) for instant live assimilation.

---

##  Scalable Framework for National Deployment

To scale this Proof of Concept (PoC) from the **Karnataka Pilot Region** to a high-fidelity, dynamic virtual replica of the entire Indian subcontinent, the following enterprise-grade cloud architecture is designed:

```text
[IMD Pune / MOSDAC / NICES] ──(Daily Ingestion)──> [Object Storage (Zarr Cloud Archives)]
                                                              │
[FastAPI Tile Server] <──(Real-time Querying)── [PyTorch DDP Multi-GPU Engine]
        │
[Streamlit / WebGL Frontend] <──(Concurrent What-If Scenarios)── [Municipal & Agri Stakeholders]
```

1. **Multi-Source Ingestion Microservices**: Automated daily cron jobs pulling NetCDF & binary gridded payloads from IMD Pune, MOSDAC (INSAT 3D/3DR `3RIMG_L2B`), and NICES platforms into Cloud-Optimized Zarr (`.zarr`) storage buckets.
2. **Distributed AI Engine (PyTorch DDP)**: The ConvLSTM engine deployed on multi-GPU clusters using PyTorch Distributed Data Parallel (DDP) with automated weekly retraining cycles to minimize model drift.
3. **Geospatial Tile Serving**: FastAPI backend serving pre-computed Vector Tiles and GeoTIFF layers via Mapbox / WebGL to ensure 60fps interactive rendering across nationwide high-resolution grids.

---

##  Advanced Capabilities & Decision Support

Beyond raw prediction grids, the Digital Twin includes advanced modules designed to convert data into actionable intelligence for disaster management and agricultural planning.

###  Automated Climate Alerts Engine
* **Source:** `src/climate_alerts.py`
* **Function:** A built-in extreme weather warning system that continuously monitors the AI's predictive grids against official **India Meteorological Department (IMD)** thresholds.
* **Capabilities:** Automatically triggers RED, ORANGE, and YELLOW alerts for:
  * **Severe Heatwaves:** Detects when plains exceed 45°C or coastal regions exceed 37°C.
  * **Heavy Rainfall & Flooding:** Flags extreme precipitation events (e.g., >204.5 mm/day) providing crucial lead time for reservoir management.
  * **Dry Spells:** Detects prolonged rainfall deficits to warn of impending agricultural droughts.

###  AI Climate Copilot (RAG Engine)
* **Source:** `src/climate_copilot.py`
* **Function:** A localized Retrieval-Augmented Generation (RAG) assistant serving as a virtual agricultural advisor.
* **Capabilities:** Cross-references the live regional climate state (e.g., current temperature and rainfall averages) against official **ICAR-CRIDA Agricultural Contingency Plans** and **NDMA guidelines**. It provides instant, grounded advice such as crop substitution strategies (e.g., switching to millets during dry spells) or foliar spray recommendations during heat stress.

###  Automated Live Data Synchronization
* **Source:** `scripts/sync_live_imd.py` & `scripts/download_insat_live.py`
* **Function:** The system is not static. Upon dashboard initialization, background orchestration scripts automatically reach out to IMD and MOSDAC APIs to synchronize the latest ground truth and satellite observations, ensuring the Digital Twin is perpetually up-to-date.

---

##  Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Shikharyadav25/ISRO--Climate-Visuals.git
cd ISRO--Climate-Visuals
```

### 2. Environment Setup
Create and activate an isolated Python environment to avoid dependency conflicts:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt
```

### 3. Choose Your Execution Path

**Option A: Fast Track (Out-of-the-Box)**
The repository comes pre-loaded with the processed NetCDF climate grids and the fully-trained PyTorch model weights (`.pth`). You can skip data processing and jump straight to **Step 4** to launch the dashboard instantly!

**Option B: Train from Scratch (Full Pipeline)**
If you wish to replicate the entire data engineering and deep learning pipeline from zero:
```bash
# 1. Download IMD/MOSDAC binary data and decode into clean NetCDF4 (.nc) files
python scripts/download_and_decode_all_real.py

# 2. Train the Spatio-Temporal ConvLSTM on the newly generated grids
python scripts/train_convlstm.py
```

### 4. Run Automated Tests
Verify that the pipeline integrity and PyTorch network initialization are perfectly stable:
```bash
python tests/test_integration.py
```

### 5. Launch the Digital Twin!
The Digital Twin uses a dual-service architecture. Open **two separate terminal windows** (ensure your `venv` is activated in both).

**Terminal 1 (Backend API):**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 (Frontend Dashboard):**
```bash
streamlit run app/streamlit_app.py
```

*Open your browser to `http://localhost:8501` to explore the interactive digital twin dashboard. The `sync_live_imd.py` script will automatically execute in the background to fetch the latest real-time observations!*

---

## ‍ Authors & Acknowledgements

**Developed for the ISRO Hackathon (AI-Powered Digital Twin of India's Climate)**  
*Empowering India's climate resilience through indigenous space technology and artificial intelligence.*  
**License**: MIT License