import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def create_temporal_features(df):
        """Add temporal features"""
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        return df
    
    @staticmethod
    def create_seasonal_features(df):
        """Add seasonal features"""
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Summer'
            elif month in [6, 7, 8, 9]:
                return 'Monsoon'
            else:
                return 'Post-Monsoon'
        
        df['season'] = df['month'].apply(get_season)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df
    
    @staticmethod
    def create_rolling_features(df):
        """Add rolling averages"""
        df['rainfall_7day_avg'] = df['rainfall'].rolling(window=7).mean()
        df['rainfall_30day_avg'] = df['rainfall'].rolling(window=30).mean()
        df['temp_diff'] = df['max_temp'] - df['min_temp']
        return df
    
    @staticmethod
    def create_lag_features(df, lags=[1, 7, 30]):
        """Add lagged features"""
        for lag in lags:
            df[f'rainfall_lag_{lag}'] = df['rainfall'].shift(lag)
            df[f'max_temp_lag_{lag}'] = df['max_temp'].shift(lag)
            df[f'min_temp_lag_{lag}'] = df['min_temp'].shift(lag)
        return df.dropna()
    
    @staticmethod
    def engineer_all_features(df):
        """Apply all feature engineering"""
        df = FeatureEngineer.create_temporal_features(df)
        df = FeatureEngineer.create_seasonal_features(df)
        df = FeatureEngineer.create_rolling_features(df)
        df = FeatureEngineer.create_lag_features(df)
        return df