import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def linear_beta_schedule(timesteps):
    """Create a linear schedule for noise variance (beta)"""
    beta_start = 0.0001  # Starting noise level
    beta_end = 0.02  # Ending noise level
    return torch.linspace(beta_start, beta_end, timesteps)

T = 200  # total diffusion steps
betas = linear_beta_schedule(T)  # Noise schedule for each timestep

alphas = 1. - betas  # Signal preservation coefficients
alphas_cumprod = torch.cumprod(alphas, axis=0)  # Cumulative product for forward diffusion

def forward_diffusion_sample(x_0, t, noise=None):
    """
    Add noise to the image x_0 at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_0)  # Generate random noise if not provided
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]  # Signal scaling factor
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]  # Noise scaling factor
    return sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise, noise  # Combine image and noise

class SimpleModel(nn.Module):
    """Simple U-Net-like model to predict noise in noisy images"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),  # First conv layer: 1->32 channels
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),  # Middle conv layer: 32->32 channels
            nn.Conv2d(32, 1, 3, padding=1),  # Output conv layer: 32->1 channel (noise prediction)
        )

    def forward(self, x, t):
        return self.net(x)  # Predict noise (t parameter unused in this simple model)

def get_data():
    """Load and preprocess MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor [0,1]
        lambda x: x * 2 - 1     # Normalize to [-1,1] for diffusion model
    ])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)  # Load MNIST training set
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # Create data loader with batching
    return dataloader

def train(model, dataloader, optimizer, epochs=5):
    """Train the diffusion model to predict noise"""
    for epoch in range(epochs):
        for step, (x, _) in enumerate(dataloader):
            x = x.to(device)  # Move images to device (CPU/GPU)
            t = torch.randint(0, T, (x.shape[0],), device=device).long()  # Random timesteps for each image
            x_noisy, noise = forward_diffusion_sample(x, t)  # Add noise to images
            noise_pred = model(x_noisy, t)  # Predict the added noise
            loss = F.mse_loss(noise_pred, noise)  # Compute MSE loss between predicted and actual noise

            optimizer.zero_grad()  # Clear gradients from previous step
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

        print(f"Epoch {epoch}: Loss {loss.item():.4f}")  # Print training progress

@torch.no_grad()
def sample(model, image_size, num_samples):
    """Generate new images by reversing the diffusion process"""
    model.eval()  # Set model to evaluation mode
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)  # Start with pure noise
    for t in reversed(range(T)):  # Reverse diffusion: T-1 -> 0
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)  # Create timestep tensor
        pred_noise = model(x, t_tensor)  # Predict noise at current timestep
        alpha = alphas[t]  # Current alpha value
        alpha_bar = alphas_cumprod[t]  # Cumulative alpha product
        beta = betas[t]  # Current beta (noise) value
        if t > 0:
            noise = torch.randn_like(x)  # Add noise for stochastic sampling
        else:
            noise = 0  # No noise at final step
        # Denoise: remove predicted noise and add small amount of noise
        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        ) + torch.sqrt(beta) * noise
    return x  # Return generated images

device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
model = SimpleModel().to(device)  # Initialize model and move to device
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer with learning rate 0.001
dataloader = get_data()  # Load MNIST dataset

train(model, dataloader, optimizer)  # Train the model

samples = sample(model, 28, 16)  # Generate 16 samples of 28x28 images
grid = torchvision.utils.make_grid(samples.cpu(), nrow=4, normalize=True)  # Create image grid (4 columns)
plt.imshow(grid.permute(1, 2, 0))  # Display grid (reorder dimensions for matplotlib)
plt.axis("off")  # Hide axes
plt.show()  # Show the generated images