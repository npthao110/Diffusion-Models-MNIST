"""
Minimal DDPM on MNIST (improved):
- Adds timestep conditioning (sinusoidal time embedding)
- Moves schedules to the correct device
- Uses DDPM posterior variance during sampling (less "scratchy" samples)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ---------------------------
# Schedules + utilities
# ---------------------------
def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)

def extract(a, t, x_shape):
    """
    a: (T,) tensor
    t: (B,) int64 tensor
    returns (B,1,1,1) broadcastable to x_shape
    """
    B = t.shape[0]
    out = a.gather(0, t)  # (B,)
    return out.view(B, *([1] * (len(x_shape) - 1)))

def make_schedules(T, device):
    betas = linear_beta_schedule(T).to(device)               # (T,)
    alphas = 1.0 - betas                                     # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)            # (T,)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # DDPM posterior variance: beta_t * (1 - abar_{t-1}) / (1 - abar_t)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }


def q_sample(x0, t, schedules, noise=None):
    """
    Forward diffusion: x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = extract(schedules["sqrt_alphas_cumprod"], t, x0.shape)
    sqrt_1mab = extract(schedules["sqrt_one_minus_alphas_cumprod"], t, x0.shape)
    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise


# ---------------------------
# Time embedding + model
# ---------------------------
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) int64
        returns: (B, dim)
        """
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=device))
        freqs = 1.0 / freqs
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class TimeConditionedCNN(nn.Module):
    """
    Deeper + wider CNN with timestep conditioning.
    Predicts eps_theta(x_t, t).
    """
    def __init__(self, in_ch=1, base_ch=128, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, base_ch),
        )

        self.b1 = Block(in_ch, base_ch)
        self.b2 = Block(base_ch, base_ch)
        self.b3 = Block(base_ch, base_ch)
        self.b4 = Block(base_ch, base_ch)
        self.b5 = Block(base_ch, base_ch)
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        h = self.b1(x)

        # timestep embedding as channel-wise bias
        temb = self.time_mlp(t)[:, :, None, None]  # (B, base_ch, 1, 1)
        h = h + temb

        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        return self.out(h)


# ---------------------------
# Sampling (reverse process)
# ---------------------------
@torch.no_grad()
def p_sample(model, x_t, t, schedules):
    """
    One reverse step:
      mean = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-abar_t)*eps_theta)
      x_{t-1} = mean + sqrt(posterior_variance_t) * z   (if t>0)
    """
    betas_t = extract(schedules["betas"], t, x_t.shape)
    sqrt_recip_alphas_t = extract(schedules["sqrt_recip_alphas"], t, x_t.shape)
    abar_t = extract(schedules["alphas_cumprod"], t, x_t.shape)
    sqrt_1m_abar_t = torch.sqrt(1.0 - abar_t)

    eps_pred = model(x_t, t)

    model_mean = sqrt_recip_alphas_t * (x_t - (betas_t / sqrt_1m_abar_t) * eps_pred)

    posterior_var_t = extract(schedules["posterior_variance"], t, x_t.shape)

    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (len(x_t.shape) - 1)))
    x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
    return x_prev

@torch.no_grad()
def sample(model, schedules, T, image_size=28, num_samples=16, device="cpu"):
    model.eval()
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)  # x_T
    for step in reversed(range(T)):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedules)
    return x


# ---------------------------
# Data + training
# ---------------------------
def to_minus1_plus1(x):
    return x * 2 - 1

def get_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_minus1_plus1),  # picklable
    ])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # Windows-friendly defaults:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # safest on Windows
        pin_memory=use_cuda
    )


def train(model, dataloader, schedules, optimizer, T, epochs=30, log_every=200, device="cpu"):
    for epoch in range(epochs):
        running = 0.0
        for step, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)

            t = torch.randint(0, T, (x0.shape[0],), device=device, dtype=torch.long)
            x_t, noise = q_sample(x0, t, schedules)

            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            if (step + 1) % log_every == 0:
                print(f"Epoch {epoch} Step {step+1}: loss={running/log_every:.4f}")
                running = 0.0

        print(f"Epoch {epoch}: last_batch_loss={loss.item():.4f}")


def show_samples(samples, nrow=4, title="DDPM samples"):
    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2  # to [0, 1]
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=nrow)
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    T = 1000
    epochs = 120
    batch_size = 128
    lr = 2e-4

    schedules = make_schedules(T, device)
    model = TimeConditionedCNN(base_ch=128, time_dim=256).to(device)

    dataloader = get_data(batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train(model, dataloader, schedules, optimizer, T, epochs=epochs, device=device)

    samples = sample(model, schedules, T, image_size=28, num_samples=16, device=device)
    show_samples(samples, nrow=4, title=f"DDPM samples (T={T}, epochs={epochs})")


if __name__ == "__main__":
    main()
