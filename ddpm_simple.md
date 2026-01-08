# Code Explanation for ddpm_simple.py

This document provides a detailed explanation of each code block in the DDPM (Denoising Diffusion Probabilistic Model) implementation for MNIST.

---

## 1. Imports (Lines 9-19)
Standard Python libraries for:
- Math operations and system utilities
- PyTorch for neural networks and tensor operations
- Torchvision for dataset loading and image utilities
- Matplotlib for visualization

---

## 2. Diffusion Schedules & Utilities (Lines 22-70)

### `linear_beta_schedule` (Lines 25-26)
**Purpose**: Creates a linear noise schedule (β values)
- Parameters: `T` (number of timesteps), `beta_start`, `beta_end`
- Returns: Tensor of β values that increase linearly from `beta_start` to `beta_end`
- **Why**: Controls how much noise is added at each timestep during forward diffusion

### `extract` (Lines 28-36)
**Purpose**: Extracts timestep-specific values from schedule tensors
- Takes a tensor `a` of shape (T,) and timestep indices `t` of shape (B,)
- Returns values reshaped to (B,1,1,1) for broadcasting to image tensors
- **Why**: Needed to apply timestep-specific coefficients to batches of images

### `make_schedules` (Lines 38-58)
**Purpose**: Precomputes all diffusion schedule coefficients
- Computes:
  - `betas`: Noise schedule (β_t)
  - `alphas`: 1 - β_t
  - `alphas_cumprod`: Cumulative product of alphas (ᾱ_t)
  - `alphas_cumprod_prev`: Previous timestep's cumulative product
  - `sqrt_alphas_cumprod`: √(ᾱ_t) - used in forward diffusion
  - `sqrt_one_minus_alphas_cumprod`: √(1 - ᾱ_t) - used in forward diffusion
  - `posterior_variance`: Variance for reverse diffusion sampling
- **Why**: Precomputation saves time during training/sampling

### `q_sample` (Lines 61-70)
**Purpose**: Forward diffusion process - adds noise to clean images
- Formula: `x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε`
- Where `ε` is random noise
- **Why**: This is how we create training data by progressively noising images

---

## 3. Time Embedding (Lines 76-89)

### `SinusoidalPosEmb` (Lines 76-89)
**Purpose**: Creates positional embeddings for timesteps
- Uses sinusoidal functions (sin/cos) with different frequencies
- Converts timestep integers into dense vector representations
- **Why**: Neural networks need timestep information to know which step of diffusion they're processing

---

## 4. U-Net Architecture (Lines 95-226)

### `_group_norm` (Lines 95-98)
**Purpose**: Helper function for Group Normalization
- Creates GroupNorm layer, falls back to single group if channels aren't divisible
- **Why**: Normalization helps training stability

### `ResBlock` (Lines 100-123)
**Purpose**: Residual block with time conditioning
- Architecture:
  1. Normalize → Activate → Convolve
  2. Add time embedding (tells block which timestep it's processing)
  3. Normalize → Activate → Dropout → Convolve
  4. Add skip connection
- **Why**: Residual blocks enable deeper networks; time conditioning allows the model to adapt behavior based on diffusion timestep

### `Downsample` (Lines 125-131)
**Purpose**: Reduces spatial resolution by 2x using strided convolution
- Uses 4x4 kernel with stride=2
- **Why**: U-Net architecture needs downsampling in the encoder path

### `Upsample` (Lines 133-140)
**Purpose**: Increases spatial resolution by 2x using interpolation + convolution
- Uses nearest-neighbor upsampling followed by convolution
- **Why**: U-Net architecture needs upsampling in the decoder path

### `TinyUNet` (Lines 143-226)
**Purpose**: Complete U-Net architecture for noise prediction
- **Structure**:
  - Time embedding MLP: Converts timestep to embedding
  - Initial convolution: Projects input to base channels
  - **Down path**: Multiple ResBlocks + Downsample layers (captures context)
  - **Mid blocks**: Two ResBlocks at bottleneck
  - **Up path**: Multiple ResBlocks + Upsample layers with skip connections (reconstructs detail)
  - Output: Final normalization, activation, convolution to output channels
- **Why**: U-Net's encoder-decoder structure with skip connections is ideal for pixel-level prediction tasks like noise estimation

---

## 5. DDPM Sampling (Lines 232-263)

### `p_sample_ddpm` (Lines 233-254)
**Purpose**: Single reverse diffusion step (denoising)
- Process:
  1. Get model's noise prediction: `ε_θ(x_t, t)`
  2. Predict clean image: `x̂_0 = (x_t - √(1-ᾱ_t)*ε) / √(ᾱ_t)`
  3. Compute posterior mean (weighted combination of x̂_0 and x_t)
  4. Sample x_{t-1} from Gaussian with this mean and posterior variance
  5. At t=0, don't add noise
- **Why**: This is the reverse diffusion process that gradually removes noise

### `sample_ddpm` (Lines 257-263)
**Purpose**: Full image generation loop
- Starts with pure noise: `x_T ~ N(0, I)`
- Iteratively calls `p_sample_ddpm` for t = T-1, T-2, ..., 1, 0
- **Why**: Generates new images by reversing the diffusion process

---

## 6. Data & Training (Lines 269-311)

### `to_minus1_plus1` (Lines 269-270)
**Purpose**: Converts pixel values from [0, 1] to [-1, 1]
- **Why**: Standard normalization for diffusion models

### `get_data` (Lines 272-285)
**Purpose**: Creates DataLoader for MNIST dataset
- Downloads MNIST if needed
- Applies transforms (to tensor, normalize to [-1, 1])
- Returns DataLoader with specified batch size
- **Why**: Provides batches of training images

### `train` (Lines 287-311)
**Purpose**: Training loop for DDPM
- For each batch:
  1. Randomly sample timesteps `t` for each image
  2. Add noise: `x_t, noise = q_sample(x_0, t)`
  3. Predict noise: `noise_pred = model(x_t, t)`
  4. Compute MSE loss: `||noise_pred - noise||²`
  5. Backpropagate and update model
- **Why**: Teaches the model to predict the noise that was added, which enables reverse diffusion

---

## 7. Visualization (Lines 316-338)

### `show_samples` (Lines 316-338)
**Purpose**: Saves and displays generated images
- Clamps values to [-1, 1] range
- Converts back to [0, 1] for display
- Creates image grid using torchvision
- Saves to file and optionally displays
- **Why**: Visual inspection of generated samples

---

## 8. Main Function (Lines 344-385)

**Purpose**: Orchestrates the entire pipeline
- Sets up device (CPU/GPU)
- Defines hyperparameters (T=1000, epochs, batch size, learning rate)
- Creates diffusion schedules
- Initializes TinyUNet model
- Creates data loader and optimizer
- Trains the model
- Generates samples and saves them
- **Why**: Entry point that runs the complete training and generation pipeline

---

## 9. Entry Point (Lines 388-389)

Runs `main()` when script is executed directly.

---

## Key Concepts Summary

1. **Forward Diffusion**: Gradually adds noise to images (q_sample)
2. **Reverse Diffusion**: Gradually removes noise to generate images (p_sample_ddpm)
3. **Training**: Model learns to predict the noise that was added
4. **U-Net**: Architecture that processes noisy images at different scales to predict noise
5. **Time Conditioning**: Model knows which timestep it's processing via embeddings

