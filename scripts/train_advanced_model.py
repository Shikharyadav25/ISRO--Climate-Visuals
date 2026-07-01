"""
Advanced Spatio-Temporal Foundation Model Training Pipeline
Demonstrates scaling the Attention-Augmented ConvLSTM across 120-years of IMD data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.pytorch_convlstm import SpatioTemporalConvLSTM, GriddedClimateDataset, train_climate_twin

def main():
    print("--- Initializing Foundation Model Training Pipeline ---")
    print("WARNING: This script is configured for A100/H100 GPU clusters.")
    print("Running locally will use subset CPU mode.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # In production, this would point to the massive 1901-2023 dataset
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    rain_path = os.path.join(data_dir, 'IMD_Gridded_Rainfall_0.25_Real_v2.nc')
    
    if not os.path.exists(rain_path):
        print(f"Error: Dataset not found at {rain_path}. Please run download_and_decode_all_real.py first.")
        return

    # Initialize Dataset
    print(f"Loading National Gridded Rainfall Data: {rain_path}")
    dataset = GriddedClimateDataset(netcdf_path=rain_path, var_name='rainfall', seq_len=30)
    
    # DataLoader optimized for distributed training
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Initialize Attention-Augmented Architecture
    # High capacity dimensions for national scale
    model = SpatioTemporalConvLSTM(
        input_dim=1, 
        hidden_dim=[128, 64, 32], 
        kernel_size=(3, 3), 
        num_layers=3
    ).to(device)
    
    print(f"Initialized SpatioTemporalConvLSTM with Spatial Attention.")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Kick off training
    train_climate_twin(
        model=model,
        dataloader=dataloader,
        epochs=100, # Extended epochs for foundation model
        lr=1e-4,
        device=device
    )

if __name__ == "__main__":
    main()
