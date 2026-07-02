# PyTorch Spatio-Temporal ConvLSTM Anomaly Model Card

This model card documents the specifications, architecture, preprocessing routines, and training parameters for the PyTorch Spatio-Temporal Convolutional LSTM (ConvLSTM) networks deployed in the climate forecasting pipeline.

---

## Model Specifications

The framework trains two independent ConvLSTM instances: one for daily precipitation anomalies and one for maximum/minimum daily temperature anomalies.

| Model Variable | Network Architecture | Input Tensor Shape | Output Tensor Shape | Conv Kernel Size | Conv Padding | Channels |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Rainfall Anomaly | 2-Layer ConvLSTM | `[B, 10, 1, 129, 135]` | `[B, 7, 1, 129, 135]` | 3 × 3 | 1 (Same) | [64, 32] |
| Temp Anomaly | 2-Layer ConvLSTM | `[B, 10, 1, 31, 31]` | `[B, 7, 2, 31, 31]` | 3 × 3 | 1 (Same) | [64, 32] |

---

## Neural Network Architecture

The network replaces standard fully connected layers in the Long Short-Term Memory cell with 2D convolutions. The equations governing the hidden state transformations at cell step $t$ are:

$$i_t = \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + W_{ci} \circ C_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + W_{cf} \circ C_{t-1} + b_f)$$
$$C_t = f_t \circ C_{t-1} + i_t \circ \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c)$$
$$o_t = \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + W_{co} \circ C_t + b_o)$$
$$H_t = o_t \circ \tanh(C_t)$$

where:
* $*$ denotes a 2D convolution operation.
* $\circ$ denotes the Hadamard (element-wise) product.
* $X_t$ is the input grid, $H_t$ is the hidden state grid, and $C_t$ is the cell state grid.
* $\sigma$ is the sigmoid activation function.

---

## Data Preprocessing and Normalization

Due to the heavy tail and skewness of raw precipitation grids, anomaly metrics are scaled before feeding them to the network to stabilize backpropagation:

### 1. Rainfall Anomaly Transformation
Anomalies are log-transformed and scaled:
$$y = \text{sign}(x) \cdot \frac{\log(1 + |x|)}{3.0}$$
During inference, predictions are mapped back to physical precipitation units (mm/day) using the inverse mapping:
$$x = \text{sign}(y) \cdot \left(e^{3.0 \cdot |y|} - 1.0\right)$$

### 2. Temperature Anomaly Transformation
Temperature anomalies are scaled by a standard factor:
$$y = \frac{x}{10.0}$$
And inverted back to Celsius via:
$$x = y \cdot 10.0$$

---

## Training Hyperparameters

The model is trained from scratch using historical records from 2015 to 2021:
* **Optimizer:** Adam (Adaptive Moment Estimation).
* **Learning Rate:** $\eta = 0.001$, decaying by a factor of 0.5 every 20 epochs.
* **Loss Function:** Mean Squared Error (MSE) loss calculated on valid grid points (excluding NaNs):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^{M} (y_i - \hat{y}_i)^2$$
* **Batch Size:** 8 sequence samples.
* **Regularization:** Spatial Dropout (rate 0.1) after each ConvLSTM layer.

---

## Uncertainty Quantification (Monte Carlo Dropout)

To estimate spatial prediction variance, the model runs with active Dropout layers during inference:
1. The system executes $N = 5$ stochastic forward passes for each forecast step.
2. It computes the voxel-wise mean grid ($\mu$) and standard deviation grid ($\sigma$):
   $$\sigma = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (\hat{y}_n - \mu)^2}$$
3. The standard deviation bounds are plotted in the GIS dashboard as uncertainty corridors (±1σ).
