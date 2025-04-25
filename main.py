import os
import time
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error
)

from joblib import Parallel, delayed
from typing import Dict, Tuple, List
from scipy.signal import lfilter
# Always work relative to the location of the script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # Set working directory to where the script is
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Saving to:", os.path.abspath("results"))
# Setup environment
os.environ['PYTHONWARNINGS'] = 'ignore'
start_time = time.time()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
# CUDA performance boost
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Ensure results directory exists
os.makedirs('results', exist_ok=True)
os.environ['PYTHONWARNINGS'] = 'ignore'
start_time = time.time()
# Updated imports for metrics and PCA
from typing import Dict, Tuple, List # Keep for potential future use, though Dict isn't used now
from scipy.signal import lfilter # Added for pink noise generation
from torch.amp import autocast, GradScaler # Added for mixed precision training

# --- Configuration ---
N_EEG_CHANNELS = 3
N_ECG_CHANNELS = 3
N_AUDIO_CHANNELS = 2
N_FREQ_FILTERS = 32 # Number of frequency features to select (per channel, x2 for mag/phase)
CONV_FILTERS_1 = 48 # Filter count for SCOFNA
CONV_FILTERS_2 = 96 # Filter count for SCOFNA
CONV_KERNEL_SIZE = 5 # Kernel size for SCOFNA
HIDDEN_SIZE = 192 # Hidden layer size for all models
OUTPUT_SIZE = 2 # Default output size, will be updated based on data
EPOCHS = 150
BATCH_SIZE = 4096
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.02
PATIENCE = EPOCHS
USE_RESIDUALS = True # Flag for SCOFNA's ConvBlocks
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# --- Utility Functions ---
# calculate_band_power function removed as requested

# --- FFT Layer (Handles Multi-channel) ---


# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# --- Utility Functions ---
# calculate_band_power function removed as requested

# --- FFT Layer (Handles Multi-channel) ---
class FFTLayerMultiChannel(nn.Module):
    """ Applies FFT, returns selected features (magnitude and phase) and frequency bins """
    def __init__(self, n_filters=N_FREQ_FILTERS, trainable_frequencies=True):
        super(FFTLayerMultiChannel, self).__init__()
        self.n_filters = n_filters # Target number of frequency bins to select features from
        self.trainable_frequencies = trainable_frequencies
        # Initialize frequency_weights as a parameter if trainable
        # We need to know the number of frequency bins first, so initialize in forward pass
        self.frequency_weights = None

    def forward(self, x, sampling_rate: float):
        epsilon = 1e-8
        batch_size, n_channels, n_timesteps = x.shape

        # Prevent FFT of zero-length input
        if n_timesteps <= 0: # Changed check to <= 0
             print("Warning: FFTLayer received zero or negative length input.")
             feat_dim = min(self.n_filters, 1) * 2; feat_dim = max(2, feat_dim)
             zeros_out = torch.zeros(batch_size, n_channels, feat_dim, device=x.device)
             dummy_mags = torch.zeros(batch_size, n_channels, 1, device=x.device)
             dummy_freqs = torch.tensor([0.0], device=x.device)
             return zeros_out, dummy_mags, dummy_freqs

        # Add small noise before FFT for numerical stability if needed, but can distort signal
        # x = x + torch.randn_like(x) * epsilon
        fft_result = torch.fft.rfft(x + epsilon, dim=2, n=n_timesteps) # Ensure FFT length matches input
        magnitudes = torch.abs(fft_result)
        phases = torch.angle(fft_result)

        # Calculate frequencies corresponding to the rfft output bins
        freqs = torch.fft.rfftfreq(n_timesteps, d=1./sampling_rate, device=x.device)
        # Ensure freqs length matches the frequency dimension of magnitudes
        actual_freq_bins = magnitudes.shape[2]
        if len(freqs) > actual_freq_bins:
            freqs = freqs[:actual_freq_bins]
        # If fewer frequencies than fft output bins (e.g. n mismatch), adjust magnitude/phase view
        elif len(freqs) < actual_freq_bins:
            magnitudes = magnitudes[:,:,:len(freqs)]
            phases = phases[:,:,:len(freqs)]
            actual_freq_bins = len(freqs) # Update actual_freq_bins


        # Handle case where rfft output is unexpectedly empty or has zero frequency bins
        if actual_freq_bins <= 0: # Changed check to <= 0
             print("Warning: FFTLayer frequency_bins is zero or negative after rfft.")
             feat_dim = min(self.n_filters, 1) * 2; feat_dim = max(2, feat_dim)
             zeros_out = torch.zeros(batch_size, n_channels, feat_dim, device=x.device)
             # Return the (now known to be empty/invalid) magnitudes and freqs
             return zeros_out, magnitudes, freqs

        # Determine the number of filters to actually use (cannot exceed available bins)
        n_filters_actual = min(self.n_filters, actual_freq_bins)

        if self.trainable_frequencies:
            # Initialize or resize weights if necessary (first pass or change in input size)
            if self.frequency_weights is None or self.frequency_weights.shape[0] != actual_freq_bins:
                 init_weights = torch.ones(actual_freq_bins, device=x.device)
                 self.frequency_weights = nn.Parameter(init_weights)
                 print(f"Initialized/Resized FFTLayer frequency weights to size: {actual_freq_bins}")

            # Clamp weights to prevent extreme values during softmax
            with torch.no_grad():
                self.frequency_weights.clamp_(-20, 20)

            # Calculate importance scores using softmax
            importance = torch.softmax(self.frequency_weights, dim=0)

            # Select top k indices based on importance
            k = min(n_filters_actual, len(importance)) # Ensure k is valid
            if k <= 0:
                print(f"Warning: FFTLayer calculated k={k}. Falling back.")
                top_indices = torch.arange(min(1, actual_freq_bins), device=x.device) # Select at least one if possible
            else:
                 _, top_indices = torch.topk(importance, k)
                 top_indices, _ = torch.sort(top_indices) # Sort indices for consistency
        else:
             # Select the first k frequency bins if not trainable
             k = n_filters_actual
             if k <= 0:
                  top_indices = torch.tensor([], dtype=torch.long, device=x.device)
             else:
                 top_indices = torch.arange(k, device=x.device)


        # Double-check indices are within bounds
        top_indices = top_indices[top_indices < actual_freq_bins]

        # Handle case where no valid indices are selected
        if len(top_indices) == 0:
             print(f"Warning: FFTLayer returning zeros - no valid indices selected (k={k}, actual_bins={actual_freq_bins}).")
             feat_dim = 2 # Minimal mag + phase feature
             zeros_out = torch.zeros(batch_size, n_channels, feat_dim, device=x.device)
             # Still return original magnitudes and frequencies (might be empty, but needed structure)
             return zeros_out, magnitudes, freqs

        # Select the corresponding magnitudes and phases using the chosen indices
        selected_magnitudes = torch.index_select(magnitudes, 2, top_indices)
        selected_phases = torch.index_select(phases, 2, top_indices)

        # Concatenate magnitudes and phases to form the frequency features
        # Shape: (batch_size, n_channels, n_filters_actual * 2)
        frequency_features = torch.cat([selected_magnitudes, selected_phases], dim=2)

        # Check for NaNs or Infs, replace them
        if torch.isnan(frequency_features).any() or torch.isinf(frequency_features).any():
             print("Warning: NaN or Inf detected in FFTLayer output! Replacing with zeros/large numbers.")
             frequency_features = torch.nan_to_num(frequency_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Return the selected features, and the *original* full magnitudes/freqs (needed for SCOFNA maybe, though bands removed)
        # Returning original magnitudes/freqs doesn't hurt FENS_MLP
        return frequency_features, magnitudes, freqs


# --- Spatial Coherence Layer (Unchanged) ---
class SpatialCoherenceLayer(nn.Module):
    """
    Applies a 1x1 convolution across channels of frequency features.
    Optionally includes a simple self-attention mechanism across channels.
    """
    def __init__(self, n_channels, n_features_per_channel, use_attention=False):
        super(SpatialCoherenceLayer, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features_per_channel
        self.use_attention = use_attention

        self.conv1x1 = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(n_channels)

        if self.use_attention:
            attn_hidden_dim = max(1, n_features_per_channel // 2) # Ensure positive dimension
            self.query_proj = nn.Linear(n_features_per_channel, attn_hidden_dim)
            self.key_proj = nn.Linear(n_features_per_channel, attn_hidden_dim)
            self.value_proj = nn.Linear(n_features_per_channel, n_features_per_channel)
            self.scale = math.sqrt(attn_hidden_dim) if attn_hidden_dim > 0 else 1
            self.attn_norm = nn.LayerNorm(n_features_per_channel)


    def forward(self, x_freq_channels):
        # Input x_freq_channels shape: (batch_size, n_channels, n_frequency_features)

        x_spatial = self.conv1x1(x_freq_channels)
        x_spatial = self.bn(x_spatial)
        if torch.isnan(x_spatial).any() or torch.isinf(x_spatial).any():
             print("Warning: NaN/Inf after SpatialCoherence BatchNorm! Replacing.")
             x_spatial = torch.nan_to_num(x_spatial)
        x_spatial = F.relu(x_spatial)


        if self.use_attention:
             if self.n_features <= 0: # Cannot perform attention with zero features
                  print("Warning: Skipping spatial attention due to zero features.")
                  return x_spatial
             q = self.query_proj(x_spatial)
             k = self.key_proj(x_spatial)
             v = self.value_proj(x_spatial)
             attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
             attn_probs = torch.softmax(attn_scores, dim=-1)
             attn_output = torch.matmul(attn_probs, v)
             x_spatial = self.attn_norm(x_spatial + attn_output)

        return x_spatial

# --- Convolutional Block (Unchanged, used by SCOFNA) ---
class ConvBlock(nn.Module):
    """ A convolutional block with Conv1d, BatchNorm, ReLU, Pooling, and optional Residual connection. """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size=2, use_residual=False, final_pool_size=None):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

        if final_pool_size:
            self.pool = nn.AdaptiveAvgPool1d(final_pool_size)
        elif pool_kernel_size > 1 :
             self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        else:
             self.pool = nn.Identity()

        if use_residual and in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm1d(out_channels)
        else:
            self.residual_projection = None

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.pool(out)

        if self.use_residual:
            if self.residual_projection:
                identity = self.residual_projection(identity)
                identity = self.residual_bn(identity)
            # Need to ensure pooling is applied to identity only if it was applied to main path
            if not isinstance(self.pool, nn.Identity):
                 identity = self.pool(identity)
            # Check shapes before adding (important with padding='same' and pooling)
            if identity.shape == out.shape:
                out = out + identity
                # out = F.relu(out) # Optional activation after residual add
            else:
                # This might happen if pooling kernel/stride doesn't perfectly divide feature length
                # Or if residual projection wasn't needed but shapes still mismatch after pool
                print(f"Warning: Residual shapes mismatch in ConvBlock. Out: {out.shape}, Identity: {identity.shape}. Skipping residual connection.")


        return out

# --- FENS MLP (`FENS_MLP_V4`) ---
class FENS_MLP_V4(nn.Module):
    """
    FENS V4: FFT -> Flatten -> MLP (Standard NN on frequency features)
    Removed Conv Blocks and Band Power.
    """
    def __init__(self, n_channels, n_timesteps, hidden_size, output_size, sampling_rate,
                 n_freq_filters=N_FREQ_FILTERS): # Removed bands argument
        super(FENS_MLP_V4, self).__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_timesteps = n_timesteps # Store for estimating input size
        self.output_size = output_size # Store output size

        # FFT Layer
        self.fft_layer = FFTLayerMultiChannel(n_filters=n_freq_filters, trainable_frequencies=True)

        # Calculate expected feature dimension after FFT and flattening
        # Note: n_timesteps is used to *estimate* the number of bins initially.
        # The actual number of features depends on fft_layer's output in forward pass.
        # We define FC layers based on this estimate. A check in forward is needed.
        rfft_bins = self.n_timesteps // 2 + 1 if self.n_timesteps > 0 else 1 # Handle n_timesteps=0 case
        actual_n_filters = min(n_freq_filters, rfft_bins)
        # Ensure minimum feature count if n_timesteps is very small
        self.n_frequency_features_per_channel = max(2, actual_n_filters * 2) # Mag + Phase

        # Estimate total input size for the first FC layer
        total_flattened_size = self.n_channels * self.n_frequency_features_per_channel
        if total_flattened_size <= 0:
            # Try to recover if estimation failed but channels > 0
            if self.n_channels > 0:
                total_flattened_size = self.n_channels * 2 # Minimal fallback
                print(f"Warning FENS V4: Estimated total_flattened_size was zero or negative. Falling back to {total_flattened_size}.")
            else:
                raise ValueError(f"FENS Error: Estimated total_flattened_size is {total_flattened_size}. Check n_channels ({self.n_channels}) and estimated features ({self.n_frequency_features_per_channel}).")


        # FC Layers (MLP)
        self.fc1 = nn.Linear(total_flattened_size, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

        print(f"FENS V4 (MLP) Initialized: FFT Features/Ch: ~{self.n_frequency_features_per_channel}, Hidden: {hidden_size}, Est. Total FC Input: {total_flattened_size}")


    def forward(self, x):
        # FFT Processing
        # x shape: (batch_size, n_channels, n_timesteps)
        x_freq_feat, _, _ = self.fft_layer(x, float(self.sampling_rate)) # Ignore mags, freqs
        # x_freq_feat shape: (batch_size, n_channels, n_frequency_features_actual)

        # Check if FFT output is valid
        if x_freq_feat.shape[2] <= 0: # Changed check to <= 0
             print("Warning: Skipping MLP layers in FENS due to zero or negative length freq features.")
             # Need to return zero output matching the expected output size
             return torch.zeros(x.shape[0], self.output_size, device=x.device)

        # Flatten the frequency features across channels
        # Shape: (batch_size, n_channels * n_frequency_features_actual)
        x_combined = x_freq_feat.flatten(1)

        # Removed band power calculation and concatenation

        # FC Layers (MLP)
        # Check for NaNs/Infs before FC layers
        if torch.isnan(x_combined).any() or torch.isinf(x_combined).any():
             print("Warning: NaN/Inf detected before FC layers in FENS! Replacing.")
             x_combined = torch.nan_to_num(x_combined, nan=0.0, posinf=1e6, neginf=-1e6)

        # Check if fc1 input size matches x_combined size - critical debug step
        expected_fc1_in = self.fc1.in_features
        actual_fc1_in = x_combined.shape[1]
        if expected_fc1_in != actual_fc1_in:
             # This can happen if n_timesteps leads to different n_filters_actual than estimated
             print(f"ERROR FENS V4: Mismatch FC1 input size. Expected {expected_fc1_in} (based on initial n_timesteps={self.n_timesteps}), Got {actual_fc1_in}. Input shape: {x_freq_feat.shape}. Adjusting layer or padding needed.")
             # Simple Fallback: Return zero output to prevent crashing.
             # A better solution requires dynamic layers or ensuring consistent input features.
             return torch.zeros(x.shape[0], self.output_size, device=x.device)

        x = self.fc1(x_combined)
        # Apply BatchNorm only if batch size > 1
        if x.shape[0] > 1:
             x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Enhanced SCOFNA (`SCOFNA_Conv_V4`) ---
class SCOFNA_Conv_V4(nn.Module):
    """
    SCOFNA V4: FFT -> Spatial Coherence -> Conv Blocks (Optional Residuals) -> FC
    Removed Band Power Features.
    """
    def __init__(self, n_channels, n_timesteps, hidden_size, output_size, sampling_rate, # Removed bands
                 n_freq_filters=N_FREQ_FILTERS, conv_filters_1=CONV_FILTERS_1, conv_filters_2=CONV_FILTERS_2, kernel_size=CONV_KERNEL_SIZE, use_residual=USE_RESIDUALS, use_spatial_attention=False):
        super(SCOFNA_Conv_V4, self).__init__()
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_timesteps = n_timesteps # Store for estimation
        self.output_size = output_size # Store output size

        # FFT Layer
        self.fft_layer = FFTLayerMultiChannel(n_filters=n_freq_filters, trainable_frequencies=True)

        # Calculate expected feature dimension after FFT
        rfft_bins = self.n_timesteps // 2 + 1 if self.n_timesteps > 0 else 1
        actual_n_filters = min(n_freq_filters, rfft_bins)
        self.n_frequency_features_per_channel = max(2, actual_n_filters * 2) # Mag + Phase

        # Spatial Coherence Layer
        # Ensure features_per_channel is positive before initializing
        if self.n_frequency_features_per_channel <= 0:
             # Try fallback if channels > 0
            if self.n_channels > 0:
                self.n_frequency_features_per_channel = 2 # Minimal fallback
                print(f"Warning SCOFNA V4: Calculated n_frequency_features_per_channel was zero or negative. Falling back to {self.n_frequency_features_per_channel}.")
            else:
                raise ValueError(f"SCOFNA Error: Cannot initialize SpatialCoherenceLayer with non-positive n_frequency_features_per_channel ({self.n_frequency_features_per_channel}). Check n_timesteps ({self.n_timesteps}).")
        self.spatial_coherence = SpatialCoherenceLayer(n_channels, self.n_frequency_features_per_channel, use_attention=use_spatial_attention)

        # Conv Block 1 (Input channels = n_channels after spatial coherence)
        self.conv_block1 = ConvBlock(n_channels, conv_filters_1, kernel_size=kernel_size, pool_kernel_size=2, use_residual=use_residual)

        # Conv Block 2 (Input channels = conv_filters_1)
        self.conv_block2 = ConvBlock(conv_filters_1, conv_filters_2, kernel_size=kernel_size, final_pool_size=1, use_residual=use_residual)

        # FC Layers
        flattened_conv_size = conv_filters_2 * 1 # Output size from AdaptiveAvgPool1d(1)
        # flattened_band_power_size removed
        total_flattened_size = flattened_conv_size # Only conv features input to FC
        if total_flattened_size <= 0:
             # Fallback if conv_filters_2 is bad
             if conv_filters_2 > 0:
                 total_flattened_size = conv_filters_2
                 print(f"Warning SCOFNA V4: Estimated FC input size was zero or negative. Falling back to {total_flattened_size}.")
             else:
                raise ValueError(f"SCOFNA Error: Estimated FC input size is {total_flattened_size}. Check conv_filters_2 ({conv_filters_2}).")


        self.fc1 = nn.Linear(total_flattened_size, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

        print(f"SCOFNA V4 Initialized: FFT Features/Ch: ~{self.n_frequency_features_per_channel}, SpatialAttn: {use_spatial_attention}, ConvFilters: {conv_filters_1}/{conv_filters_2}, Hidden: {hidden_size}, Est. Total FC Input: {total_flattened_size}, Residuals: {use_residual}")

    def forward(self, x):
        # FFT Processing
        x_freq_feat, _, _ = self.fft_layer(x, float(self.sampling_rate)) # Ignore mags, freqs
        # x_freq_feat shape: (batch_size, n_channels, n_frequency_features_actual)

        # Check FFT output validity before spatial coherence
        if x_freq_feat.shape[2] <= 0: # Changed check to <= 0
            print("Warning: Skipping Spatial Coherence and Conv layers in SCOFNA due to zero or negative length freq features.")
            # Output must match expected shape for FC layer
            x_conv_flat = torch.zeros(x.shape[0], CONV_FILTERS_2, device=x.device)
        else:
            # Spatial Coherence
            # Input shape: (batch_size, n_channels, n_frequency_features)
            # Ensure input feature dimension matches what SpatialCoherenceLayer expects
            sc_input_features = x_freq_feat.shape[2]
            if sc_input_features != self.spatial_coherence.n_features:
                 print(f"Warning SCOFNA V4: Spatial Coherence input feature mismatch. Expected {self.spatial_coherence.n_features}, Got {sc_input_features}. Skipping spatial/conv.")
                 # Fallback: zero output matching expected conv output size
                 x_conv_flat = torch.zeros(x.shape[0], CONV_FILTERS_2, device=x.device)
            else:
                x_sc = self.spatial_coherence(x_freq_feat)
                # Output shape: (batch_size, n_channels, n_frequency_features)

                # Conv Blocks on spatially processed features
                if x_sc.shape[2] <= 0: # Double check after spatial coherence
                    print("Warning: Skipping Conv layers in SCOFNA due to zero or negative length features after spatial coherence.")
                    x_conv_flat = torch.zeros(x.shape[0], CONV_FILTERS_2, device=x.device)
                else:
                    # Input to conv1 has n_channels channels, feature dim = n_freq_features
                    x_c1 = self.conv_block1(x_sc)
                    # Input to conv2 has conv_filters_1 channels
                    x_c2 = self.conv_block2(x_c1)
                    # x_c2 shape: (batch_size, conv_filters_2, 1)
                    x_conv_flat = x_c2.flatten(1) # Flatten features: (batch_size, conv_filters_2)

        # Removed Band Power Features calculation and concatenation
        x_combined = x_conv_flat

        # FC Layers
        if torch.isnan(x_combined).any() or torch.isinf(x_combined).any():
             print("Warning: NaN/Inf detected before FC layers in SCOFNA! Replacing.")
             x_combined = torch.nan_to_num(x_combined, nan=0.0, posinf=1e6, neginf=-1e6)

        # Check FC1 input size match
        expected_fc1_in = self.fc1.in_features
        actual_fc1_in = x_combined.shape[1]
        if expected_fc1_in != actual_fc1_in:
             print(f"ERROR SCOFNA V4: Mismatch FC1 input size. Expected {expected_fc1_in}, Got {actual_fc1_in}. ConvFlat shape: {x_conv_flat.shape}")
             # Fallback: Return zero output
             return torch.zeros(x.shape[0], self.output_size, device=x.device)

        x = self.fc1(x_combined)
        # Apply BatchNorm only if batch size > 1
        if x.shape[0] > 1:
             x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- Standard NN Architecture (Baseline - Unchanged) ---
class StandardNN_Simple_BN_V2(nn.Module):
    """ A slightly deeper/wider Standard NN for a stronger baseline. """
    def __init__(self, input_size, hidden_size, output_size):
        super(StandardNN_Simple_BN_V2, self).__init__()
        # Ensure input_size is positive
        if input_size <= 0:
             # Try fallback
             input_size = 10 # Arbitrary small positive number
             print(f"Warning SNN V2: input_size was zero or negative. Falling back to {input_size}")
             # raise ValueError(f"SNN Error: input_size must be positive, got {input_size}") # Original behaviour

        self.fc1 = nn.Linear(input_size, hidden_size * 3)
        self.bn1 = nn.BatchNorm1d(hidden_size * 3)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.output_size = output_size # Store output size

        print(f"SNN V2 Initialized: Input: {input_size}, HiddenSizes: {hidden_size*3}/{hidden_size*2}/{hidden_size}, Output: {output_size}")


    def forward(self, x):
        # Input x shape: (batch_size, input_size_flattened)
        if x.shape[1] != self.fc1.in_features:
             print(f"ERROR SNN V2: Input feature mismatch. Expected {self.fc1.in_features}, Got {x.shape[1]}")
             return torch.zeros(x.shape[0], self.output_size, device=x.device) # Zero output

        x = torch.nan_to_num(x)

        x = self.fc1(x)
        if x.shape[0] > 1: x = self.bn1(x)
        x = F.relu(x); x = self.dropout1(x)

        x = self.fc2(x)
        if x.shape[0] > 1: x = self.bn2(x)
        x = F.relu(x); x = self.dropout2(x)

        x = self.fc3(x)
        if x.shape[0] > 1: x = self.bn3(x)
        x = F.relu(x); x = self.dropout3(x)

        x = self.fc4(x)
        return x

# --- Device selection (CUDA if available, else CPU) ---
if torch.cuda.is_available():
    device = torch.device('cuda')
    DEVICE_NAME = 'CUDA'
else:
    device = torch.device('cpu')
    DEVICE_NAME = 'CPU'
print(f"Using device: {device}")

def set_memory_fraction(fraction: float):
    """
    If running on CUDA, set per‐process GPU memory fraction.
    fraction: float between 0 and 1.
    """
    if device.type == 'cuda':
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=device)
            print(f"Set GPU memory fraction to {fraction}")
        except Exception as e:
            print(f"Warning: Could not set memory fraction: {e}")

def train_model(model,
                X_train, y_train,
                X_val,   y_val,
                epochs=EPOCHS,
                batch_size=256,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                patience=PATIENCE,
                grad_clip_value=1.0,
                epoch_cb=None):
    model.to(device)
    scaler = GradScaler() if device.type == "cuda" else None

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.astype(int)))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val.astype(int)))

    num_workers = min(4, os.cpu_count())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=(device.type=='cuda'),
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=(device.type=='cuda'), prefetch_factor=2)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs, val_rmses, val_maes = [], [], [], [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, batches = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(Xb)
                loss = criterion(out, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                optimizer.step()
            running_loss += loss.item()
            batches += 1

        train_losses.append(running_loss / batches if batches > 0 else float("nan"))

        # Validation
        model.eval()
        v_loss, v_batches, correct, total = 0.0, 0, 0, 0
        y_true_batches, y_pred_batches = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                loss = criterion(out, yb)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                v_loss += loss.item()
                v_batches += 1
                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                y_true_batches.append(yb.cpu().numpy())
                y_pred_batches.append(pred.cpu().numpy())

        val_losses.append(v_loss / v_batches if v_batches > 0 else float("nan"))
        val_accs.append(correct / total if total > 0 else float("nan"))
        if len(y_true_batches) > 0:
            y_true = np.concatenate(y_true_batches)
            y_pred = np.concatenate(y_pred_batches)
            val_rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            val_maes.append(mean_absolute_error(y_true, y_pred))
        else:
            val_rmses.append(float("nan")); val_maes.append(float("nan"))

        scheduler.step()
        current_val_loss = val_losses[-1]
        if not np.isnan(current_val_loss) and current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch_cb:
            epoch_cb(epoch, epochs)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}: TrainLoss={train_losses[-1]:.4f}, ValLoss={val_losses[-1]:.4f}, ValAcc={val_accs[-1]:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "val_rmse": val_rmses,
        "val_mae": val_maes
    }

def evaluate_model(model, X_test, y_test, batch_size=BATCH_SIZE):
    """ Evaluates the model and returns accuracy, RMSE, MAE, confusion matrix, predictions, and labels. """
    model.to(device)
    model.eval()

    # Determine default num_classes based on model's final layer
    num_classes = OUTPUT_SIZE # Default fallback
    if hasattr(model, 'output_size'):
        num_classes = model.output_size
    elif hasattr(model, 'fc2') and hasattr(model.fc2, 'out_features'):
        num_classes = model.fc2.out_features
    elif hasattr(model, 'fc4') and hasattr(model.fc4, 'out_features'):
        num_classes = model.fc4.out_features

    if np.isnan(X_test).any() or np.isinf(X_test).any():
        print(f"ERROR: NaNs/Infs found in TEST data for {model.__class__.__name__}. Evaluation cannot proceed.")
        return 0.0, float('nan'), float('nan'), np.zeros((num_classes, num_classes), dtype=int), [], np.arange(num_classes)

    try:
        # Ensure data has compatible types
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test.astype(int))
        test_dataset = TensorDataset(X_test_t, y_test_t)
        num_workers = min(4, os.cpu_count())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
    except Exception as e:
        print(f"Error creating test DataLoader: {e}")
        return 0.0, float('nan'), float('nan'), np.zeros((num_classes, num_classes), dtype=int), [], np.arange(num_classes)

    y_pred_list = []
    y_true_list = []
    correct = 0
    total = 0
    nan_detected_eval = False
    test_batch_count = 0

    with torch.no_grad():
        for k, (X_batch_test, y_batch_test) in enumerate(test_loader):
            if torch.isnan(X_batch_test).any() or torch.isinf(X_batch_test).any():
                 # print(f"Warning: NaN/Inf detected in test batch {k}. Skipping batch.") # Verbose
                 continue
            X_batch_test, y_batch_test = X_batch_test.to(device), y_batch_test.to(device)

            try:
                 test_outputs = model(X_batch_test)
            except Exception as e:
                 print(f"ERROR during model evaluation forward pass (batch {k}): {e}")
                 nan_detected_eval = True; break

            if nan_detected_eval: break

            if torch.isnan(test_outputs).any() or torch.isinf(test_outputs).any():
                print(f"ERROR: NaN/Inf output detected during evaluation (batch {k}). Aborting.")
                nan_detected_eval = True; break

            _, predicted = torch.max(test_outputs.data, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(y_batch_test.cpu().numpy())
            total += y_batch_test.size(0)
            correct += (predicted == y_batch_test).sum().item()
            test_batch_count += 1

    print(f"\n--- Evaluation: {model.__class__.__name__} ---")

    # Use true labels from the test set to determine unique labels if prediction list is empty
    unique_labels = np.unique(y_true_list) if y_true_list else np.unique(y_test.astype(int))
    if len(unique_labels) == 0: # If still no labels (e.g., y_test was empty)
        unique_labels = np.arange(num_classes) # Fallback to range based on output size
    target_names = [f"Class {int(i)}" for i in unique_labels] # Ensure labels are ints for naming


    if nan_detected_eval or test_batch_count == 0 or total == 0 or not y_true_list:
        print("Evaluation failed, produced NaN/Inf, processed no valid batches, or had no true labels.")
        test_accuracy = 0.0
        rmse = float('nan')
        mae = float('nan')
        conf_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        class_report = "Classification Report: N/A (Evaluation Failed)"
    else:
        test_accuracy = correct / total
        conf_matrix = confusion_matrix(y_true_list, y_pred_list, labels=unique_labels)
        # Calculate RMSE and MAE
        rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
        mae = mean_absolute_error(y_true_list, y_pred_list)
        try:
            class_report = classification_report(y_true_list, y_pred_list, zero_division=0, labels=unique_labels, target_names=target_names)
        except ValueError as e:
            print(f"Warning: Could not generate classification report: {e}")
            class_report = f"Classification Report: Error ({e})"

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    #print(class_report)
    # Ensure labels returned match the ones used in the matrix/report
    return test_accuracy, rmse, mae, conf_matrix, y_pred_list, unique_labels


def plot_comparison(histories, labels, dataset_name,
                    metrics=('train_loss', 'val_loss', 'val_acc', 'val_rmse', 'val_mae')):
    metric_map = {
        'train_loss': 'Training Loss',
        'val_loss':   'Validation Loss',
        'val_acc':    'Validation Accuracy',
        'val_rmse':   'Validation RMSE',
        'val_mae':    'Validation MAE'
    }

    plt.figure(figsize=(12, 4 * len(metrics)))
    max_epochs = max(len(hist.get('train_loss', [])) for hist in histories)

    for i, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, i)
        for hist, label in zip(histories, labels):
            values = hist.get(metric, [])
            if not values: continue
            values = values + [np.nan] * (max_epochs - len(values))  # pad
            plt.plot(range(1, max_epochs + 1), values[:max_epochs], label=f'{label} {metric_map[metric]}')

        plt.title(f"{dataset_name} - {metric_map.get(metric, metric)}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_map.get(metric, metric))
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_training_comparison_full.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

def plot_fft_spectrum(signal: np.ndarray, sampling_rate: int, signal_name: str, results_dir: str):
    """
    Plots FFT magnitude spectrum for the given signal (shape: [channels, timesteps])
    and saves it as a PNG file.
    """
    fft_freqs = np.fft.rfftfreq(signal.shape[1], d=1./sampling_rate)
    plt.figure(figsize=(12, 6))
    for ch in range(signal.shape[0]):
        fft_vals = np.abs(np.fft.rfft(signal[ch]))
        plt.plot(fft_freqs, fft_vals, label=f'Channel {ch+1}', linewidth=2.0)

    plt.title(f'{signal_name} - FFT Magnitude Spectrum', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.8)
    plt.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"{signal_name}_FFT_Spectrum.png")
    try:
        plt.savefig(save_path, dpi=300)
        print(f"FFT spectrum saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save FFT plot for {signal_name}: {e}")
    plt.close()

def plot_eeg_like_stack(X_data, y_data, dataset_name, n_examples=3):
    """
    Stacked plot of signals: rows = (example × channel), colored by class.
    Removed per‑subplot titles and labels.
    """
    if X_data is None or X_data.size == 0:
        print(f"Skipping stacked plot for {dataset_name}: No data.")
        return

    n_samples, n_channels, n_timesteps = X_data.shape
    n_examples = min(n_examples, n_samples)
    time_axis = np.arange(n_timesteps)

    fig, axes = plt.subplots(n_examples * n_channels, 1,
                             figsize=(12, 2.5 * n_examples * n_channels),
                             sharex=True)

    # flatten axes array
    axes = np.array(axes).flatten()

    for ex in range(n_examples):
        label = int(y_data[ex]) if y_data is not None else 0
        color = 'red' if label == 1 else 'blue'
        for ch in range(n_channels):
            idx = ex * n_channels + ch
            ax = axes[idx]
            ax.plot(time_axis, X_data[ex, ch, :], color=color, linewidth=1)
            ax.set_ylabel(f'Ch{ch+1}', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(0, n_timesteps)
            ax.tick_params(axis='both', labelsize=8)

    plt.xlabel('Time Step', fontsize=10)
    fig.suptitle(f'{dataset_name} – Stacked Signals', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(RESULTS_DIR, f"{dataset_name}_stacked_signals.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved stacked plot: {out_path}")


def plot_confusion_matrix(conf_matrix, classes, title, dataset_name, model_name):
    filename = f'results/{dataset_name}_{model_name}_confusion.png'
    plt.figure(figsize=(6, 5))

    # Ensure classes are hashable (e.g., ints or strings) for mapping later if needed
    try:
        classes_hashable = [int(c) for c in classes] # Prefer ints if possible
    except (ValueError, TypeError):
        classes_hashable = [str(c) for c in classes] # Fallback to strings


    if isinstance(conf_matrix, list): conf_matrix = np.array(conf_matrix)

    # Validate matrix shape against number of classes
    if conf_matrix is None or not isinstance(conf_matrix, np.ndarray) or conf_matrix.shape != (len(classes_hashable), len(classes_hashable)):
         print(f"Skipping confusion matrix for {title}: Invalid matrix shape {conf_matrix.shape if conf_matrix is not None else 'None'} vs expected ({len(classes_hashable)}, {len(classes_hashable)}) for labels {classes_hashable}")
         plt.text(0.5, 0.5, 'Evaluation Failed\nor Matrix Invalid', ha='center', va='center', fontsize=12)
         plt.xticks([]); plt.yticks([])
    else:
         # Use string representation of classes for tick labels
         class_labels_str = [str(c) for c in classes_hashable]
         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_labels_str, yticklabels=class_labels_str, annot_kws={"size": 12})
         plt.ylabel('True Label', fontsize=12)
         plt.xlabel('Predicted Label', fontsize=12)
         plt.xticks(rotation=0); plt.yticks(rotation=0)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{dataset_name}_training_comparison.png"))
    plt.close()
    print(f"Saved confusion matrix: {filename}")


# --- NEW Plotting Function: Training Data Scatter via PCA ---
def plot_training_scatter(X_train, y_train, dataset_name):
    """ Plots a scatter of first two PCA components of the training data. """
    out_file = f"{dataset_name}_training_scatter_pca.png"
    out_path = os.path.join(RESULTS_DIR, out_file)
    print(f"Generating PCA scatter plot for {dataset_name} training data...")

    try:
        if X_train is None or X_train.size == 0 or y_train is None or y_train.size == 0:
            print(f"Skipping scatter plot for {dataset_name}: No data.")
            return
        if X_train.shape[0] != y_train.shape[0]:
            print(f"Skipping scatter plot for {dataset_name}: Sample/label mismatch.")
            return

        # flatten
        n_samples = X_train.shape[0]
        X_flat = X_train.reshape(n_samples, -1) if X_train.ndim == 3 else X_train

        # sanitize
        X_flat = np.nan_to_num(X_flat)
        # PCA
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_flat)
        pca = PCA(n_components=2); X_pca = pca.fit_transform(X_scaled)

        # plot
        plt.figure(figsize=(8,6))
        labels = np.unique(y_train)
        colors = plt.cm.viridis(np.linspace(0,1,len(labels)))
        for lbl, clr in zip(labels, colors):
            idx = np.where(y_train==lbl)
            plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f"Class {int(lbl)}", alpha=0.6, color=clr, s=15)
        plt.title(f"{dataset_name} Training Data (PCA 1 vs 2)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved scatter plot: {out_path}")

    except Exception as e:
        print(f"Error in plot_training_scatter for {dataset_name}: {e}")
        # don't re-raise, prevents crashing


# --- Data Generation Functions (Unchanged) ---
def scale_multi_channel_data(X_train, X_val, X_test):
    if X_train is None or X_val is None or X_test is None: return None, None, None, None
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0: # Check for empty arrays
         print("Warning: Empty data array passed to scaling function.")
         return X_train, X_val, X_test, None
    n_train_samples, n_channels, n_timesteps = X_train.shape
    if n_channels <= 0 or n_timesteps <= 0: # Check for invalid dimensions
        print(f"Warning: Invalid dimensions for scaling ({n_channels} channels, {n_timesteps} timesteps). Returning unscaled data.")
        return X_train, X_val, X_test, None

    scalers = [StandardScaler() for _ in range(n_channels)]
    X_train_scaled, X_val_scaled, X_test_scaled = np.zeros_like(X_train), np.zeros_like(X_val), np.zeros_like(X_test)
    print("Scaling data per channel...")
    try:
        for c in range(n_channels):
             # Check variance before fitting
             # Ensure data is 2D for variance check (samples * timesteps)
             channel_data_train = X_train[:, c, :].reshape(-1, 1)
             if channel_data_train.size == 0 or np.var(channel_data_train) < 1e-8:
                  print(f"Warning: Channel {c} in training data has near-zero variance or is empty. Skipping scaling for this channel.")
                  X_train_scaled[:, c, :] = X_train[:, c, :]
                  X_val_scaled[:, c, :] = X_val[:, c, :]
                  X_test_scaled[:, c, :] = X_test[:, c, :]
                  scalers[c] = None # Mark scaler as unused
                  continue

             # Fit on reshaped (samples * timesteps, 1) data for StandardScaler
             # But transform requires (samples, timesteps) shape
             scalers[c].fit(X_train[:, c, :])
             X_train_scaled[:, c, :] = scalers[c].transform(X_train[:, c, :])
             X_val_scaled[:, c, :] = scalers[c].transform(X_val[:, c, :])
             X_test_scaled[:, c, :] = scalers[c].transform(X_test[:, c, :])
    except ValueError as e:
        print(f"ERROR during scaling channel {c}: {e}. Returning unscaled data.")
        return X_train, X_val, X_test, None

    X_train_scaled = np.nan_to_num(X_train_scaled); X_val_scaled = np.nan_to_num(X_val_scaled); X_test_scaled = np.nan_to_num(X_test_scaled)
    print("Scaling complete.")
    return X_train_scaled, X_val_scaled, X_test_scaled, scalers

def generate_eeg_artifacts(n_timesteps, sampling_rate):
    artifacts = np.zeros(n_timesteps)
    if n_timesteps <= 0 or sampling_rate <=0: return artifacts # Guard against invalid input
    t = np.linspace(0, n_timesteps / sampling_rate, n_timesteps, endpoint=False)
    # Ensure start index calculation avoids negative values if duration > n_timesteps
    blink_dur_samples = int(np.random.uniform(0.2, 0.5) * sampling_rate)
    blink_dur_samples = min(max(1, blink_dur_samples), n_timesteps) # Ensure valid duration
    if n_timesteps > blink_dur_samples and np.random.rand() < 0.1: # Eye Blink
        blink_start = np.random.randint(0, max(1, n_timesteps - blink_dur_samples)) # Ensure range is valid
        blink_end = min(n_timesteps, blink_start + blink_dur_samples)
        blink_amp = np.random.uniform(3, 6)
        if blink_end > blink_start: # Check for valid range before linspace/sin
            blink_time = np.linspace(0, np.pi, blink_end - blink_start)
            artifacts[blink_start:blink_end] += blink_amp * np.sin(blink_time) * (1 + np.random.randn()*0.1)

    muscle_dur_samples = int(np.random.uniform(0.5, 1.5) * sampling_rate)
    muscle_dur_samples = min(max(1, muscle_dur_samples), n_timesteps) # Ensure valid duration
    if n_timesteps > muscle_dur_samples and np.random.rand() < 0.2: # Muscle Artifact
        muscle_start = np.random.randint(0, max(1, n_timesteps - muscle_dur_samples)) # Ensure range is valid
        muscle_end = min(n_timesteps, muscle_start + muscle_dur_samples)
        muscle_amp = np.random.uniform(0.5, 1.5)
        muscle_freq = np.random.uniform(20, 50)
        if muscle_end > muscle_start: # Check for valid range before slicing/sin
            artifacts[muscle_start:muscle_end] += muscle_amp * np.sin(2 * np.pi * muscle_freq * t[muscle_start:muscle_end] + np.random.rand()*2*np.pi) * np.random.randn(muscle_end - muscle_start) * 0.5
    return artifacts

def load_eeg_data_multi_channel(n_samples=1000, n_timesteps=256, n_channels=N_EEG_CHANNELS):
    print(f"\nGenerating Synthetic Multi-Channel EEG Data V4 ({n_channels} channels, {n_timesteps} timesteps)...")
    if n_samples <= 0 or n_timesteps <= 0 or n_channels <= 0:
         print("Error: Invalid dimensions requested for EEG data generation.")
         return None, None, None, None, None, None, 0.0
    np.random.seed(42)
    X = np.zeros((n_samples, n_timesteps, n_channels)); y = np.zeros(n_samples)
    sampling_rate = 128.0; noise_std = 0.20
    t = np.linspace(0, n_timesteps / sampling_rate, n_timesteps, endpoint=False)
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)} # Internal use only
    channel_locs = np.linspace(-1, 1, n_channels) if n_channels > 1 else np.array([0.0])
    alpha_spatial = np.exp(-(channel_locs - 0.7)**2 / (2 * 0.4**2)); beta_spatial = np.exp(-(channel_locs - 0.0)**2 / (2 * 0.5**2))
    theta_spatial = np.exp(-(channel_locs - 0.2)**2 / (2 * 0.6**2)); delta_spatial = np.ones(n_channels) * 0.5
    # Normalize spatial profiles safely
    alpha_spatial /= (alpha_spatial.sum()+1e-8); beta_spatial /= (beta_spatial.sum()+1e-8); theta_spatial /= (theta_spatial.sum()+1e-8); delta_spatial /= (delta_spatial.sum()+1e-8)

    for i in range(n_samples):
        class_id = i % 2; y[i] = class_id; signal = np.zeros((n_timesteps, n_channels))
        if class_id == 0: alpha_p, beta_p, theta_p = (1.2, 0.45, 0.35); delta_p, gamma_p = (0.2, 0.15)
        else: alpha_p, beta_p, theta_p = (0.55, 1.05, 0.8); delta_p, gamma_p = (0.2, 0.15)
        alpha_p*=np.random.uniform(0.8,1.2); beta_p*=np.random.uniform(0.8,1.2); theta_p*=np.random.uniform(0.8,1.2); delta_p*=np.random.uniform(0.8,1.2); gamma_p*=np.random.uniform(0.8,1.2)

        for c in range(n_channels):
            s_a = alpha_spatial[c]*alpha_p*np.sin(2*np.pi*np.random.uniform(*bands['alpha'])*t+np.random.rand()*2*np.pi)
            s_b = beta_spatial[c]*beta_p*np.sin(2*np.pi*np.random.uniform(*bands['beta'])*t+np.random.rand()*2*np.pi)
            s_th= theta_spatial[c]*theta_p*np.sin(2*np.pi*np.random.uniform(*bands['theta'])*t+np.random.rand()*2*np.pi)
            s_d = delta_spatial[c]*delta_p*np.sin(2*np.pi*np.random.uniform(*bands['delta'])*t+np.random.rand()*2*np.pi)
            s_g = 0.1*gamma_p*np.sin(2*np.pi*np.random.uniform(*bands['gamma'])*t+np.random.rand()*2*np.pi)
            signal[:, c] = s_a + s_b + s_th + s_d + s_g

        common_noise = noise_std*np.random.randn(n_timesteps)*0.3; indep_noise = noise_std*np.random.randn(n_timesteps, n_channels)*0.7
        artifacts = generate_eeg_artifacts(n_timesteps, sampling_rate)
        artifact_profile = np.exp(-(channel_locs - (-0.5))**2 / (2 * 0.8**2)); artifact_profile /= (artifact_profile.sum() + 1e-8)
        signal += common_noise[:, np.newaxis] + indep_noise + artifacts[:, np.newaxis] * artifact_profile[np.newaxis, :] * 2.0
        wander_freq = np.random.uniform(0.1, 0.5); wander_amp = np.random.uniform(0.1, 0.3)
        baseline_wander = wander_amp * np.sin(2 * np.pi * wander_freq * t + np.random.rand() * 2 * np.pi)
        signal += baseline_wander[:, np.newaxis]
        X[i, :, :] = np.clip(signal, -8, 8)

    X = X.transpose(0, 2, 1) # Shape: (n_samples, n_channels, n_timesteps)
    # Ensure we have enough samples for splitting
    if n_samples < 4:
        print("Error: Not enough samples generated for train/val/test split.")
        return None, None, None, None, None, None, sampling_rate

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    except ValueError as e:
        print(f"Error during train/test split (likely too few samples per class): {e}. Returning None.")
        return None, None, None, None, None, None, sampling_rate

    X_train_s, X_val_s, X_test_s, scalers = scale_multi_channel_data(X_train, X_val, X_test)
    # If scaling failed, use original data but issue warning
    if scalers is None:
        print("Warning: Scaling failed, using unscaled EEG data.")
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
    print(f"EEG Data Shapes: Train={X_train_s.shape}, Val={X_val_s.shape}, Test={X_test_s.shape}")
    if np.isnan(X_train_s).any() or np.isnan(X_val_s).any() or np.isnan(X_test_s).any():
         print("ERROR: NaNs detected in final EEG data! Attempting to replace with 0.")
         X_train_s=np.nan_to_num(X_train_s); X_val_s=np.nan_to_num(X_val_s); X_test_s=np.nan_to_num(X_test_s)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, sampling_rate

def generate_ecg_wave(t_rel, rr_interval, wave_type='qrs', class_id=0, variability=0.05):
    wave = np.zeros_like(t_rel); rand_factor = 1.0 + np.random.normal(0, variability)
    if wave_type == 'p':
        p_peak = -0.15 * rr_interval; p_amp = (0.1 + np.random.normal(0, 0.02)) * rand_factor; p_width = (0.02 * rr_interval) * rand_factor
        wave = p_amp * np.exp(-((t_rel - p_peak)**2) / (2 * p_width**2 + 1e-9)) # Add epsilon for stability
    elif wave_type == 'qrs':
        qrs_amp = (1.0 + np.random.normal(0, 0.05)) * rand_factor; qrs_width = (0.015 * rr_interval) * rand_factor * np.random.uniform(0.9, 1.1)
        wave = qrs_amp * np.exp(-(t_rel**2) / (2 * qrs_width**2 + 1e-9)) # R
        q_amp = (0.15*qrs_amp)*rand_factor; q_peak = -0.03*rr_interval; q_width = qrs_width*0.7; wave -= q_amp * np.exp(-((t_rel-q_peak)**2)/(2*q_width**2 + 1e-9)) # Q
        s_amp = (0.20*qrs_amp)*rand_factor; s_peak = 0.04*rr_interval; s_width = qrs_width*0.8; wave -= s_amp * np.exp(-((t_rel-s_peak)**2)/(2*s_width**2 + 1e-9)) # S
    elif wave_type == 't':
        t_peak = (0.30 * rr_interval) * rand_factor; t_amp = (0.25 + np.random.normal(0, 0.03)) * rand_factor * (0.9 if class_id == 0 else 1.1)
        t_width = (0.04 * rr_interval) * rand_factor * (1.0 if class_id == 0 else 1.15)
        wave = t_amp * np.exp(-((t_rel - t_peak)**2) / (2 * t_width**2 + 1e-9))
    return wave

def simulate_rr_intervals_v2(n_beats, base_hr=75, variability=0.05, class_id=0):
    if n_beats <= 0: return np.array([]) # Handle invalid input
    base_rr = 60.0 / base_hr; t = np.zeros(n_beats); rr_intervals = np.zeros(n_beats); current_rr = base_rr * (1 + np.random.normal(0, variability*0.5))
    for k in range(n_beats):
        if k > 0: t[k] = t[k-1] + rr_intervals[k-1]
        lf_amp = 0.06*base_rr*(1.6 if class_id == 0 else 0.6); hf_amp = 0.04*base_rr*(0.7 if class_id == 0 else 1.4)
        lf_freq = np.random.uniform(0.08, 0.12); hf_freq = np.random.uniform(0.20, 0.30)
        lf_osc = lf_amp*np.sin(2*np.pi*lf_freq*t[k]+np.random.rand()*2*np.pi); hf_osc = hf_amp*np.sin(2*np.pi*hf_freq*t[k]+np.random.rand()*2*np.pi)
        noise_var = variability*base_rr*(0.8 if class_id == 0 else 1.2); noise = np.random.normal(0, noise_var)
        ar_factor = 0.1; current_rr = base_rr + lf_osc + hf_osc + noise
        if k > 0: current_rr += ar_factor * (rr_intervals[k-1] - base_rr)
        rr_intervals[k] = np.clip(current_rr, 0.35, 1.8)
    return rr_intervals

def load_ecg_data_multi_channel(n_samples=1000, n_timesteps=512, n_channels=N_ECG_CHANNELS):
    print(f"\nGenerating Synthetic Multi-Channel ECG Data V4 ({n_channels} channels, {n_timesteps} timesteps)...")
    if n_samples <= 0 or n_timesteps <= 0 or n_channels <= 0:
         print("Error: Invalid dimensions requested for ECG data generation.")
         return None, None, None, None, None, None, 0.0
    np.random.seed(42)
    X = np.zeros((n_samples, n_timesteps, n_channels)); y = np.zeros(n_samples)
    sampling_rate = 128.0; duration = n_timesteps / sampling_rate
    t_ecg = np.linspace(0, duration, n_timesteps, endpoint=False)
    # Ensure lead_vectors has enough rows for requested channels
    base_lead_vectors = np.array([[1.0, 0.1, 0.2], [0.5, 1.0, 0.6], [-0.2, 0.8, 1.0], [0.8, -0.1, 0.4], [-0.1, 0.9, -0.3]]) # Added more base leads
    if n_channels > base_lead_vectors.shape[0]:
        print(f"Warning: Requested {n_channels} ECG channels, but only {base_lead_vectors.shape[0]} base lead vectors defined. Repeating vectors.")
        repeats = math.ceil(n_channels / base_lead_vectors.shape[0])
        lead_vectors = np.tile(base_lead_vectors, (repeats, 1))[:n_channels]
    else:
        lead_vectors = base_lead_vectors[:n_channels]


    for i in range(n_samples):
        class_id = i % 2; y[i] = class_id
        est_n_beats = int(duration/(60.0/np.random.uniform(65,85)))+10; rr_intervals = simulate_rr_intervals_v2(est_n_beats, base_hr=np.random.uniform(60, 90), variability=0.06, class_id=class_id)
        if rr_intervals.size == 0: continue # Skip sample if RR interval generation failed
        beat_times = np.cumsum(rr_intervals); beat_times = beat_times[beat_times < duration + 0.5]
        if beat_times.size == 0: continue # Skip if no beats fall within duration

        sample_lead_vectors = lead_vectors + np.random.normal(0, 0.15, lead_vectors.shape)
        p_signal = np.zeros(n_timesteps); qrs_signal = np.zeros(n_timesteps); t_signal = np.zeros(n_timesteps)

        for k, beat_t in enumerate(beat_times):
             # Ensure window indices are valid
             r_peak_idx = np.argmin(np.abs(t_ecg - beat_t)); window_size = int(sampling_rate * 1.0)
             start_idx = max(0, r_peak_idx - window_size // 2); end_idx = min(n_timesteps, r_peak_idx + window_size // 2)
             # Skip if window is empty
             if start_idx >= end_idx: continue
             t_rel_window = t_ecg[start_idx:end_idx] - beat_t
             current_rr = rr_intervals[k] if k < len(rr_intervals) else (60.0/75.0)
             p_wave_win = generate_ecg_wave(t_rel_window, current_rr, 'p', class_id, variability=0.06)
             qrs_wave_win = generate_ecg_wave(t_rel_window, current_rr, 'qrs', class_id, variability=0.06)
             t_wave_win = generate_ecg_wave(t_rel_window, current_rr, 't', class_id, variability=0.06)
             p_signal[start_idx:end_idx] += p_wave_win; qrs_signal[start_idx:end_idx] += qrs_wave_win; t_signal[start_idx:end_idx] += t_wave_win

        base_components = np.stack([p_signal, qrs_signal, t_signal], axis=0)
        baseline_wander = np.zeros(n_timesteps)
        for _ in range(np.random.randint(1, 4)):
            wander_freq = np.random.uniform(0.05, 0.4); wander_amp = np.random.uniform(0.02, 0.08)
            baseline_wander += wander_amp * np.sin(2 * np.pi * wander_freq * t_ecg + np.random.rand() * 2 * np.pi)
        powerline_freq = 60.0; powerline_amp = np.random.uniform(0.01, 0.05)
        powerline_noise = powerline_amp * np.sin(2 * np.pi * powerline_freq * t_ecg + np.random.rand() * 2 * np.pi)
        common_noise_amp = 0.03; common_noise = common_noise_amp * np.random.randn(n_timesteps)

        for channel in range(n_channels):
            lead_proj = sample_lead_vectors[channel] @ base_components
            channel_noise_amp = 0.02; channel_noise = channel_noise_amp * np.random.randn(n_timesteps)
            final_signal = lead_proj + baseline_wander + powerline_noise + common_noise + channel_noise
            # Ensure artifact indices are valid
            art_len_samples = np.random.randint(20, 50)
            art_len_samples = min(max(1, art_len_samples), n_timesteps) # Ensure valid length
            if n_timesteps > art_len_samples and np.random.rand() < 0.05: # Motion artifact
                 artifact_start = np.random.randint(0, max(1, n_timesteps - art_len_samples)) # Ensure valid range
                 artifact_amp = np.random.uniform(0.5, 1.5) * (1 if np.random.rand()<0.5 else -1)
                 final_signal[artifact_start : artifact_start+art_len_samples] += artifact_amp
            X[i, :, channel] = np.clip(final_signal, -4, 4)

    X = X.transpose(0, 2, 1) # Shape: (n_samples, n_channels, n_timesteps)
    # Ensure we have enough samples for splitting
    if n_samples < 4:
        print("Error: Not enough samples generated for train/val/test split.")
        return None, None, None, None, None, None, sampling_rate

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    except ValueError as e:
        print(f"Error during train/test split (likely too few samples per class): {e}. Returning None.")
        return None, None, None, None, None, None, sampling_rate

    X_train_s, X_val_s, X_test_s, scalers = scale_multi_channel_data(X_train, X_val, X_test)
    if scalers is None:
        print("Warning: Scaling failed, using unscaled ECG data.")
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
    print(f"ECG Data Shapes: Train={X_train_s.shape}, Val={X_val_s.shape}, Test={X_test_s.shape}")
    if np.isnan(X_train_s).any() or np.isnan(X_val_s).any() or np.isnan(X_test_s).any():
         print("ERROR: NaNs detected in final ECG data! Attempting to replace with 0.");
         X_train_s=np.nan_to_num(X_train_s); X_val_s=np.nan_to_num(X_val_s); X_test_s=np.nan_to_num(X_test_s)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, sampling_rate

def generate_complex_tone_v2(t, base_freq, harmonics_amps, fm_rate=0, fm_depth=0):
    signal = np.zeros_like(t)
    # Calculate modulated frequency safely
    if fm_rate > 0 and fm_depth > 0:
        mod_freq = base_freq * (1 + fm_depth * np.sin(2 * np.pi * fm_rate * t))
    else:
        mod_freq = base_freq # Scalar if no FM

    # Ensure mod_freq is broadcastable with t if it's an array
    if isinstance(mod_freq, np.ndarray) and mod_freq.shape != t.shape:
        print(f"Warning: Mismatch shapes in generate_complex_tone. t: {t.shape}, mod_freq: {mod_freq.shape}. Using base_freq.")
        mod_freq = base_freq # Fallback to scalar

    for i, amp in enumerate(harmonics_amps):
        if amp > 1e-4:
            # Ensure multiplication happens correctly (mod_freq might be scalar or array)
            try:
                 phase = 2 * np.pi * (i + 1) * mod_freq * t + np.random.rand() * np.pi
                 signal += amp * np.sin(phase)
            except ValueError as e:
                 print(f"Error calculating phase for harmonic {i+1}: {e}. Skipping harmonic.")
                 continue # Skip this harmonic if shapes cause error
    return signal

def apply_reverb(signal, delay_samples, decay_factor):
    reverberated_signal = np.copy(signal)
    delay_samples = int(delay_samples) # Ensure integer
    # Check bounds carefully
    if delay_samples > 0 and delay_samples < len(signal):
        reverberated_signal[delay_samples:] += signal[:-delay_samples] * decay_factor
    elif delay_samples < 0 and abs(delay_samples) < len(signal): # Handle negative delay (pre-delay?) although unusual
        reverberated_signal[:delay_samples] += signal[-delay_samples:] * decay_factor
    return reverberated_signal

def load_audio_data_multi_channel(n_samples=1000, n_timesteps=1024, n_channels=N_AUDIO_CHANNELS):
    print(f"\nGenerating Realistic Multi-Channel Audio Data V4 ({n_channels} channels, {n_timesteps} timesteps)...")
    if n_samples <= 0 or n_timesteps <= 0 or n_channels <= 0:
         print("Error: Invalid dimensions requested for Audio data generation.")
         return None, None, None, None, None, None, 0.0
    np.random.seed(42)
    X = np.zeros((n_samples, n_timesteps, n_channels)); y = np.zeros(n_samples)
    sampling_rate = 8000.0; duration = n_timesteps / sampling_rate
    t = np.linspace(0, duration, n_timesteps, endpoint=False)

    def generate_pink_noise(n_steps):
        # Simplified approximation using filtered white noise
        burn_in = 100
        if n_steps <= 0: return np.array([]) # Handle zero steps
        wn = np.random.randn(n_steps + burn_in)
        # Using simple IIR filter coefficients (approx -3dB/octave)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        try:
            pink_ish = lfilter(b, a, wn)
            # Remove burn-in samples and normalize
            pink_ish = pink_ish[burn_in:]
            std_dev = np.std(pink_ish)
            if std_dev > 1e-8: # Avoid division by zero
                return pink_ish / std_dev
            else:
                return np.zeros_like(pink_ish) # Return zeros if variance is too low
        except ValueError as e:
             # This can happen if wn is too short after potential negative n_steps input
             print(f"Error filtering noise (ValueError likely due to length): {e}. Returning white noise.")
             wn_fallback = np.random.randn(n_steps)
             std_dev = np.std(wn_fallback)
             return wn_fallback / (std_dev + 1e-8)
        except Exception as e:
            print(f"Error filtering noise: {e}. Returning white noise.")
            # Fallback to white noise if filtering fails
            wn_fallback = np.random.randn(n_steps)
            std_dev = np.std(wn_fallback)
            return wn_fallback / (std_dev + 1e-8)


    for i in range(n_samples):
        class_id = i % 2; y[i] = class_id
        base_freq = np.random.uniform(150, 350); fm_rate = np.random.uniform(1, 5) if np.random.rand() < 0.3 else 0
        fm_depth = np.random.uniform(0.005, 0.02) if fm_rate > 0 else 0
        if class_id == 0: harmonics = [1.0, 0.2, 0.7, 0.15, 0.4, 0.1, 0.2]; fm_rate *= 0.5
        else: harmonics = [0.8, 0.6, 0.2, 0.5, 0.1, 0.3, 0.05]; fm_rate *= 1.2
        harmonics = [h * np.random.uniform(0.8, 1.2) for h in harmonics]
        base_signal = generate_complex_tone_v2(t, base_freq, harmonics, fm_rate, fm_depth)

        # Apply envelope safely
        attack_len = int(sampling_rate*np.random.uniform(0.01,0.05)); attack_len = min(max(0, attack_len), n_timesteps)
        decay_len = int(sampling_rate*np.random.uniform(0.1,0.3)); decay_len = min(max(0, decay_len), n_timesteps)
        envelope = np.ones(n_timesteps)
        if attack_len > 0: envelope[:attack_len] = np.linspace(0, 1, attack_len)
        # Calculate decay start and actual length carefully
        decay_start = max(attack_len, n_timesteps - decay_len)
        actual_decay_len = n_timesteps - decay_start
        if actual_decay_len > 0:
             envelope[decay_start:] = np.linspace(1, 0, actual_decay_len)
        base_signal *= envelope


        noise_type = np.random.choice(['white', 'pink'], p=[0.6, 0.4]); base_noise_level = np.random.uniform(0.03, 0.08)
        if noise_type == 'white':
            common_noise = base_noise_level * np.random.randn(n_timesteps) * 0.5
        else:
            pink_noise_generated = generate_pink_noise(n_timesteps)
            common_noise = base_noise_level * pink_noise_generated * 0.5 if pink_noise_generated.size == n_timesteps else np.zeros(n_timesteps)

        reverb_delay = int(sampling_rate*np.random.uniform(0.01,0.05)); reverb_decay = np.random.uniform(0.1, 0.4)
        base_signal_reverb = apply_reverb(base_signal, reverb_delay, reverb_decay)


        for channel in range(n_channels):
            # Pan factor determines gain difference between channels
            pan_factor = 0.7 if class_id == 0 else 0.3 # Example pan based on class
            # Simple linear panning law (adjust gain based on pan factor and channel index)
            if n_channels == 1:
                 gain_factor = 1.0
            elif channel == 0: # Left channel
                 gain_factor = pan_factor
            elif channel == n_channels - 1: # Right channel
                 gain_factor = 1.0 - pan_factor
            else: # Center channels (equal gain, adjust based on needs)
                 gain_factor = 0.5 * (pan_factor + (1.0 - pan_factor)) # Average gain for center

            channel_signal = base_signal_reverb * (gain_factor * 1.8 + 0.1) # Apply gain

            # Apply Interaural Time Difference (ITD)
            if n_channels > 1:
                # Map channel index to a range [-1, 1] where -1 is far left, 1 is far right
                itd_position = (channel / (n_channels - 1) - 0.5) * 2 if n_channels > 1 else 0
                max_delay_ms = 0.7 # Max ITD in ms
                # Calculate delay in samples based on position
                delay_samples = int(itd_position * max_delay_ms * sampling_rate / 1000 * np.random.uniform(0.5, 1.0))

                # Apply delay by padding/shifting
                if delay_samples > 0 and delay_samples < len(channel_signal): # Positive delay (pad beginning)
                     channel_signal = np.pad(channel_signal[:-delay_samples], (delay_samples, 0), 'constant')
                elif delay_samples < 0 and abs(delay_samples) < len(channel_signal): # Negative delay (pad end)
                     channel_signal = np.pad(channel_signal[abs(delay_samples):], (0, abs(delay_samples)), 'constant')
                # else: no delay applied if delay is too large or zero


            channel_noise_indep = base_noise_level * np.random.randn(n_timesteps) * 0.3 * (1.0 + gain_factor*0.2)
            signal = channel_signal + common_noise + channel_noise_indep
            if np.random.rand() < 0.03: # Clipping
                 clip_level = np.random.uniform(0.8, 1.5); signal = np.clip(signal * 1.5, -clip_level, clip_level)
            X[i, :, channel] = np.clip(signal, -3, 3) # Final clipping


        if np.isnan(X[i]).any() or np.isinf(X[i]).any():
             print(f"Warning: NaN/Inf detected in generated audio sample {i}. Replacing.")
             X[i] = np.nan_to_num(X[i])

    X = X.transpose(0, 2, 1) # Shape: (n_samples, n_channels, n_timesteps)
    # Ensure we have enough samples for splitting
    if n_samples < 4:
        print("Error: Not enough samples generated for train/val/test split.")
        return None, None, None, None, None, None, sampling_rate

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    except ValueError as e:
        print(f"Error during train/test split (likely too few samples per class): {e}. Returning None.")
        return None, None, None, None, None, None, sampling_rate

    X_train_s, X_val_s, X_test_s, scalers = scale_multi_channel_data(X_train, X_val, X_test)
    if scalers is None:
        print("Warning: Scaling failed, using unscaled Audio data.")
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test
    print(f"Audio Data Shapes: Train={X_train_s.shape}, Val={X_val_s.shape}, Test={X_test_s.shape}")
    if np.isnan(X_train_s).any() or np.isnan(X_val_s).any() or np.isnan(X_test_s).any():
         print("ERROR: NaNs detected in final Audio data! Attempting to replace with 0.");
         X_train_s=np.nan_to_num(X_train_s); X_val_s=np.nan_to_num(X_val_s); X_test_s=np.nan_to_num(X_test_s)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, sampling_rate

def run_fold(dataset_name: str,
             X_pool: np.ndarray,
             y_pool: np.ndarray,
             sampling_rate: float,
             train_idx: np.ndarray,
             test_idx: np.ndarray,
             fold: int,
             epochs: int = EPOCHS,
             epoch_cb=None) -> Dict[str, Tuple[float,float,float]]:
    """
    Runs training+evaluation on one CV fold.
    Returns a dict mapping model_name -> (acc, rmse, mae).
    """
    n_samples, n_channels, n_timesteps = X_pool.shape

    # split
    X_tr, y_tr = X_pool[train_idx], y_pool[train_idx]
    X_te, y_te = X_pool[test_idx],  y_pool[test_idx]

    # further split train→train/val for SNN
    X_tr_full, X_val_full, y_tr_full, y_val_full = train_test_split(
        X_tr.reshape(len(train_idx), -1), y_tr,
        test_size=0.2, stratify=y_tr, random_state=fold
    )

    # instantiate fresh models
    num_classes = len(np.unique(y_pool))
    models = {
      'SNN_V2':       StandardNN_Simple_BN_V2(n_channels*n_timesteps, HIDDEN_SIZE, num_classes),
      'FENS_MLP_V4':  FENS_MLP_V4(n_channels, n_timesteps, HIDDEN_SIZE, num_classes, sampling_rate),
      'SCOFNA_Conv_V4': SCOFNA_Conv_V4(n_channels, n_timesteps, HIDDEN_SIZE, num_classes, sampling_rate)
    }

    results = {}
    histories = {}
    conf_mats = {}
    for name, model in models.items():
        # select appropriate inputs
        if name.startswith('SNN'):
            X_tr_i,  y_tr_i,  X_val_i,  y_val_i  = X_tr_full,   y_tr_full,   X_val_full,   y_val_full
            X_te_i,  y_te_i  = X_te.reshape(len(test_idx), -1), y_te
        else:
            X_tr_i = X_tr.reshape(len(train_idx), n_channels, n_timesteps)
            X_val_i = X_tr_i
            X_te_i  = X_te.reshape(len(test_idx), n_channels, n_timesteps)
            y_tr_i, y_val_i, y_te_i = y_tr, y_tr, y_te

        # train & eval
        trained, history = train_model(
            model, X_tr_i, y_tr_i, X_val_i, y_val_i,
            epochs=epochs, batch_size=BATCH_SIZE,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, grad_clip_value=1.0,
            epoch_cb=epoch_cb
        )
        acc, rmse, mae, cm, _, _ = evaluate_model(trained, X_te_i, y_te_i)
        results[name] = (acc, rmse, mae)
        conf_mats[name] = cm
        histories[name] = history

    return results, histories, conf_mats
    
    
# --- Expose for runner.py ---
data_loaders = {
    "EEG": load_eeg_data_multi_channel,
    "ECG": load_ecg_data_multi_channel,
    "Audio": load_audio_data_multi_channel
}
n_splits = 5

if __name__ == "__main__":
    from window import App
    App().mainloop()

def plot_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)  # use .0f to handle floats
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
