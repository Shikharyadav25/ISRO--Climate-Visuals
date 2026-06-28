import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_loader import ModelLoader
from src.predictions import ClimatePredictor
from src.feature_engineering import FeatureEngineer

# Page config
st.set_page_config(
    page_title="India's Climate Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #FF6B35;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #004E89;
            border-bottom: 3px solid #FF6B35;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Cache models
@st.cache_resource
def load_models():
    return ModelLoader()

@st.cache_resource
def load_predictor():
    model_loader = load_models()
    return ClimatePredictor(model_loader)

# Load
model_loader = load_models()
predictor = load_predictor()

# Header
st.markdown('<h1 class="main-header">🌍 India\'s Climate Digital Twin</h1>', 
            unsafe_allow_html=True)
st.markdown("**AI-Powered Climate Prediction & Analysis using LSTM & Deep Learning**")
st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.header("📊 Navigation")
    page = st.radio(
        "Select a page:",
        ["📈 Dashboard", 
         "🔮 Predictions", 
         "🎯 What-If Simulation",
         "📊 Analysis",
         "ℹ️ About"]
    )

# PAGE 1: DASHBOARD
if page == "📈 Dashboard":
    st.markdown('<h2 class="section-header">Current Climate Dashboard</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌧️ Current Rainfall", "45.2 mm", "+2.1 mm")
    with col2:
        st.metric("🌡️ Max Temperature", "32.5°C", "-1.2°C")
    with col3:
        st.metric("❄️ Min Temperature", "22.1°C", "+0.5°C")
    with col4:
        st.metric("📍 Grid Points", "1,242", "+145")
    
    st.markdown('<h3 class="section-header">Temperature Trend</h3>', 
                unsafe_allow_html=True)
    
    # Dummy data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                          periods=30, freq='D')
    max_temps = np.random.uniform(30, 35, 30)
    min_temps = np.random.uniform(20, 25, 30)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=max_temps,
                            mode='lines+markers', name='Max Temperature',
                            line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=dates, y=min_temps,
                            mode='lines+markers', name='Min Temperature',
                            line=dict(color='blue', width=2)))
    fig.update_layout(title="30-Day Temperature Trend", height=400,
                     xaxis_title="Date", yaxis_title="Temperature (°C)")
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: PREDICTIONS
elif page == "🔮 Predictions":
    st.markdown('<h2 class="section-header">Climate Predictions</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Prediction Settings")
        pred_days = st.slider("Days ahead:", 1, 30, 7)
        
    with col2:
        st.subheader("📊 Results")
        if st.button("🎯 Generate Predictions"):
            with st.spinner("Generating predictions..."):
                # Dummy predictions
                dates = pd.date_range(start=datetime.now(), 
                                     periods=pred_days, freq='D')
                rainfall = np.random.uniform(20, 80, pred_days)
                max_temp = np.random.uniform(30, 38, pred_days)
                min_temp = np.random.uniform(18, 26, pred_days)
                
                pred_df = pd.DataFrame({
                    'Date': dates,
                    'Rainfall (mm)': rainfall,
                    'Max Temp (°C)': max_temp,
                    'Min Temp (°C)': min_temp
                })
                
                st.success(f"✓ Predictions generated for {pred_days} days")
                st.dataframe(pred_df, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Rainfall (mm)'],
                                        mode='lines+markers', name='Rainfall',
                                        line=dict(color='blue')))
                fig.update_layout(title="Rainfall Prediction", height=400)
                st.plotly_chart(fig, use_container_width=True)

# PAGE 3: WHAT-IF SIMULATION
elif page == "🎯 What-If Simulation":
    st.markdown('<h2 class="section-header">What-If Scenario Analysis</h2>', 
                unsafe_allow_html=True)
    
    st.info("🔬 Explore how changes in rainfall and temperature affect climate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Adjust Parameters")
        rainfall_change = st.slider("Rainfall Change (%)", -50, 50, 0, 5)
        temp_change = st.slider("Temperature Change (°C)", -5, 5, 0, 0.5)
    
    with col2:
        st.subheader("📈 Impact Results")
        if st.button("▶️ Run Simulation"):
            with st.spinner("Running simulation..."):
                # Sample current conditions
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
                
                results = predictor.simulate_what_if(
                    current, rainfall_change, temp_change
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Original Rainfall", 
                             f"{results['original_rainfall']:.1f} mm")
                    st.metric("Original Max Temp", 
                             f"{results['original_max_temp']:.1f}°C")
                with col_b:
                    st.metric("Modified Rainfall", 
                             f"{results['modified_rainfall']:.1f} mm",
                             delta=f"{rainfall_change:+.1f}%")
                    st.metric("Modified Max Temp", 
                             f"{results['modified_max_temp']:.1f}°C",
                             delta=f"{temp_change:+.1f}°C")
                
                st.warning("⚠️ **Impact**: Significant changes in water availability")

# PAGE 4: ANALYSIS
elif page == "📊 Analysis":
    st.markdown('<h2 class="section-header">Historical Analysis (2022-2025)</h2>', 
                unsafe_have_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        analysis = st.selectbox("Analysis Type:", 
                               ["Seasonal Trends", "Annual Comparison"])
    with col2:
        variable = st.selectbox("Variable:", 
                               ["Rainfall", "Temperature"])
    
    if st.button("📉 Analyze"):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        values = np.random.uniform(30 if variable == "Rainfall" else 20, 
                                  80 if variable == "Rainfall" else 35, 12)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=values, marker=dict(color='steelblue')))
        fig.update_layout(title=f"{variable} by Month", height=400)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 5: ABOUT
elif page == "ℹ️ About":
    st.markdown('<h2 class="section-header">About This Project</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Objectives
    
    1. **Digital Twin Development**: AI-powered representation of India's climate
    2. **Climate Predictions**: LSTM & Deep Learning for rainfall & temperature
    3. **Interactive Dashboard**: Real-time climate insights
    4. **Scenario Analysis**: What-if simulations for impact assessment
    
    ## 📊 Data & Models
    
    - **Data**: India Meteorological Department (IMD) 2022-2025
    - **Rainfall Model**: LSTM (Sequence-to-Sequence)
    - **Temperature Model**: Dense Neural Network
    - **Metrics**: RMSE, MAE, R² Score
    
    ## 🏆 Project: ISRO Hackathon Submission
    """)

st.markdown("---")
st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("🚀 ISRO Climate Digital Twin | Powered by AI/ML")