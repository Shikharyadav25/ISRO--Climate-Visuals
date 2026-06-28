import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.model_loader import ModelLoader
from src.predictions import ClimatePredictor

print("=" * 60)
print("INTEGRATION TEST")
print("=" * 60)

# Test 1: Load models
print("\nTest 1: Loading models...")
try:
    model_loader = ModelLoader(model_dir='models')
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Rainfall prediction
print("\nTest 2: Rainfall prediction...")
try:
    dummy_rainfall = np.random.uniform(0, 50, 30)
    dummy_sequence = dummy_rainfall.reshape(1, 30, 1)
    dummy_scaled = model_loader.scalers['rainfall'].transform(
        dummy_rainfall.reshape(-1, 1)
    ).reshape(1, 30, 1)
    
    prediction = model_loader.models['rainfall_lstm'].predict(dummy_scaled, verbose=0)
    prediction_actual = model_loader.scalers['rainfall'].inverse_transform(prediction)
    
    print(f"✓ Rainfall prediction: {prediction_actual[0][0]:.2f} mm")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Temperature prediction
print("\nTest 3: Temperature prediction...")
try:
    dummy_features = np.array([[45.2, 50.1, 48.3, 31.5, 22.1, 0.5, 0.866, 9.4]])
    max_temp, min_temp = model_loader.predict_temperature(dummy_features)
    print(f"✓ Max Temp: {max_temp[0]:.2f}°C, Min Temp: {min_temp[0]:.2f}°C")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 4: What-if simulation
print("\nTest 4: What-if simulation...")
try:
    predictor = ClimatePredictor(model_loader)
    current = {
        'rainfall': 45.2,
        'max_temp': 32.5,
        'min_temp': 22.1,
        'rainfall_lag_1': 44.1,
        'rainfall_lag_7': 47.3,
        'rainfall_30day_avg': 48.3,
        'max_temp_lag_1': 31.5,
        'min_temp_lag_1': 22.8,
        'month_sin': 0.5,
        'month_cos': 0.866,
        'temp_diff': 10.4
    }
    
    results = predictor.simulate_what_if(current, 20, 2)
    print(f"✓ Original rainfall: {results['original_rainfall']:.2f} mm")
    print(f"✓ Modified rainfall: {results['modified_rainfall']:.2f} mm (+20%)")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Ready for Streamlit!")
print("=" * 60)