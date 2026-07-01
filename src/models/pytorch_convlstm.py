import os
import torch
import torch.nn as nn
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader

class ConvLSTMCell(nn.Module):
    """
    Core Spatio-Temporal ConvLSTM Cell for 2D Gridded Climate Feature Extraction.
    Operates on 5D tensors: [batch, channels, height (lat), width (lon)]
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module to focus on localized extreme weather phenomena.
    Mimics the attention mechanisms used in modern foundation models (GraphCast, Pangu-Weather).
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_weights = self.sigmoid(self.conv(x))
        return x * attn_weights


class SpatioTemporalConvLSTM(nn.Module):
    """
    Attention-Augmented Multi-layer ConvLSTM Network for high-resolution gridded weather prediction.
    Architecture designed for IMD Gridded Rainfall (0.25°) and Max Temp (1.0°).
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(SpatioTemporalConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        self.attention_list = nn.ModuleList()

        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            self.cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                               hidden_dim=hidden_dim[i],
                                               kernel_size=kernel_size))
            self.attention_list.append(SpatialAttention(in_channels=hidden_dim[i]))

        # Final output projection layer to map back to target weather variable channels (e.g., 1 for rainfall)
        self.out_conv = nn.Conv2d(in_channels=hidden_dim[-1], out_channels=1, kernel_size=1)

    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: [batch_size, time_steps, channels, height, width]
        """
        batch_size, seq_len, _, height, width = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=batch_size, image_size=(height, width))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                
                # Apply Spatial Attention to focus on critical features
                h = self.attention_list[layer_idx](h)
                
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Project final layer state to predicted variable grid
        pred_grid = self.out_conv(layer_output_list[-1][:, -1, :, :, :]) # Predict next time step
        return pred_grid, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class GriddedClimateDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and windowing NetCDF gridded data (IMD / MOSDAC).
    Transforms xarray DataArrays into 5D Spatio-Temporal PyTorch Tensors.
    """
    def __init__(self, netcdf_path, var_name, seq_len=30, transform=None):
        self.ds = xr.open_dataset(netcdf_path)
        self.data_array = self.ds[var_name].values
        # Fill missing/ocean masked values with 0 for neural net processing
        self.data_array = np.nan_to_num(self.data_array, nan=0.0)
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.data_array) - self.seq_len

    def __getitem__(self, idx):
        # Input sequence of length `seq_len`
        x = self.data_array[idx : idx + self.seq_len]
        # Target is the immediate next time step grid
        y = self.data_array[idx + self.seq_len]

        # Add channel dimension: [time_steps, channels=1, lat, lon]
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_climate_twin(model, dataloader, epochs=20, lr=1e-3, device='cpu'):
    """
    Production training loop for the AI Digital Twin ConvLSTM Engine.
    Includes optimization, loss computation (MSE + MAE tracking), and checkpoint saving.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(f"--- Starting Spatio-Temporal Model Training on {device.upper()} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_mae = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            predictions, _ = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in spatio-temporal dynamics
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                total_mae += torch.mean(torch.abs(predictions - targets)).item()

        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | RMSE Loss: {np.sqrt(avg_loss):.4f} | MAE: {avg_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save trained checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/climate_twin_convlstm_final.pth')
    print("[SUCCESS] Model training complete. Final weights saved to `checkpoints/climate_twin_convlstm_final.pth`.")

if __name__ == "__main__":
    # Test script execution for architecture verification
    print("Verifying ConvLSTM Spatio-Temporal Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Mock parameters matching IMD 0.25 deg grid slice (e.g., 24x20 grid points in Karnataka)
    dummy_input = torch.randn(2, 10, 1, 24, 20).to(device) # [batch=2, time=10, channels=1, lat=24, lon=20]
    
    model = SpatioTemporalConvLSTM(input_dim=1, hidden_dim=[64, 32], kernel_size=(3, 3), num_layers=2).to(device)
    output, states = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Predicted Output Grid Shape: {output.shape}")
    print("[SUCCESS] Spatio-Temporal ConvLSTM initialized and validated successfully!")
