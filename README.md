# AI-Powered Digital Twin of India's Climate

## 🎯 Project Overview

An ISRO Hackathon project using India's National Meteorological Data to create an AI-powered digital twin for climate prediction and scenario analysis.

**Technologies**: TensorFlow, LSTM, Streamlit, Python

---

## 📊 Quick Start

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Run Tests
```bash
python tests/test_integration.py
```

### 3. Launch App
```bash
cd app
streamlit run streamlit_app.py
```

Open browser → `http://localhost:8501`

---

## 📁 Project Structure
limate_digital_twin/
├── models/                    # Trained models
├── data/processed/           # Processed data
├── src/                      # Python modules
│   ├── model_loader.py
│   ├── predictions.py
│   ├── feature_engineering.py
│   └── data_preprocessing.py
├── app/streamlit_app.py     # Frontend
├── tests/test_integration.py
└── requirements.txt


---

## 🤖 Models

### Rainfall LSTM
- Input: Last 30 days rainfall
- Output: Next day rainfall prediction
- Metrics: RMSE ~8.5mm, R² > 0.78

### Temperature DNN
- Input: 8 features (lagged values, seasonal, temporal)
- Output: Max & Min temperature
- Metrics: RMSE ~1.2°C, R² > 0.75

---

## 📈 Results

| Metric | Rainfall | Max Temp | Min Temp |
|--------|----------|----------|----------|
| RMSE   | 8.45 mm  | 1.25°C   | 0.98°C   |
| MAE    | 6.12 mm  | 0.87°C   | 0.65°C   |
| R²     | 0.7823   | 0.7654   | 0.8145   |

---

## 🌾 Use Cases

1. **Agriculture**: Crop planning & yield prediction
2. **Water Resources**: Monsoon forecasting & reservoir management
3. **Disaster Management**: Flood/drought early warning
4. **Urban Planning**: Climate-aware infrastructure design

---

## 📚 Data Sources

- Rainfall: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html
- Temperature: https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
- Period: 2022-2025 (4 years)
- Resolution: 0.25° × 0.25° (Rainfall), 1° × 1° (Temperature)

---

## 👨‍💼 Author & License

**ISRO Hackathon 2024-25**
MIT License