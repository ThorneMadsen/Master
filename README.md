# MLPVAE: Multi-Layer Perceptron Variational Autoencoder

This repository contains a Conditional Variational Autoencoder (CVAE) implementation using Multi-Layer Perceptron (MLP) blocks for energy imbalance prediction and analysis.

## Overview

The MLPVAE model is designed to predict energy imbalance patterns using conditional inputs. It uses a probabilistic approach to generate predictions with uncertainty quantification through percentile-based metrics.

## Requirements

### Python Dependencies

Make sure you have Python 3.7+ installed. Install the required packages:

### Required Packages:
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `seaborn` - Statistical plotting
- `matplotlib` - Plotting
- `plotly` - Interactive plotting
- `jupyter` - Jupyter notebook support
- `scipy` - Scientific computing

## Data Requirements

The model expects data in the following format:

1. **Main data file**: `data/X2.npy` 
   - Shape: `(N, F, T)` where:
     - `N` = number of samples
     - `F` = number of features
     - `T` = time steps (24 hours)
   - The first feature (`X[:, 0, :]`) is treated as the target (imbalance)
   - Remaining features (`X[:, 1:, :]`) are used as conditions

2. **Data structure**:
   - Target: Energy imbalance values (shape: `N x 24`)
   - Conditions: Input features like weather, prices, etc. (shape: `N x (F-1) x 24`)

## Usage

The notebook is organized into several sections:

#### Data Loading and Preprocessing
```python
# Load and preprocess data
X = np.load("../data/X2.npy")
X = X[:855]  # Use first 855 samples
N, F, T = X.shape

target = X[:, 0, :]      # Imbalance data
condition = X[:, 1:, :]  # Conditioning variables
```

#### Model Architecture

The model consists of:
- **MLPBlock**: Gated MLP with layer normalization and residual connections
- **Encoder**: Encodes condition + target to latent space (μ, σ)
- **Decoder**: Decodes latent + condition to target prediction (μ, σ)
- **CVAE**: Complete conditional VAE combining encoder and decoder

#### Training Configuration

Key hyperparameters:
```python
latent_dim = 32          # Latent space dimensionality
hidden_dim = 48          # Hidden layer size
mlp_blocks = 4           # Number of MLP blocks
dropout_prob = 0.4       # Dropout probability
learning_rate = 5e-5     # Learning rate
batch_size = 64          # Batch size
num_epochs = 2000        # Training epochs
```

#### Training the Model

# Initialize model
model = CVAE(cond_dim, target_dim, latent_dim, mlp_blocks=4, hidden_dim=48, dropout_prob=0.4)


### 3. Model Evaluation

The notebook includes comprehensive evaluation metrics:

#### Directional Accuracy Metrics
- **PIC (Percentile Interval Coverage)**: How often targets exceed predicted percentiles
- **DMAE (Directional Mean Absolute Error)**: Absolute error for directional predictions

#### Visualization Functions
- `plot_val_with_std_band()`: Show uncertainty bands around predictions
- `calculate_average_metrics()`: Compute average performance metrics
- `plot_percentile_accuracy()`: Evaluate percentile prediction accuracy
- `plot_masking_impact()`: Analyze impact of partial information

# Calculate average performance metrics
results = calculate_average_metrics(model, X, val_dataset, latent_dim, num_samples=1000)