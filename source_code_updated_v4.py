# -*- coding: utf-8 -*-
"""
Minimal DDPM on MNIST (improved v4 - Tiny U-Net):
✅ Tiny U-Net backbone (big jump in sample quality vs plain CNN)
✅ x0 clamp during sampling (stability / fewer artifacts)
✅ EMA weights for cleaner samples
✅ Improved DDIM sampling (clean integer timestep schedule + stable update)
✅ Saves ablation grids into ./output_v4 (won't overwrite v2/v3 outputs)

Run:
  python source_code_updated_v4.py
"""

import math
import copy
import os
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

    # DDPM posterior variance: beta_t * (1 - abar_{t-1}) / (1 - abar_t)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
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
# Time embedding
# ---------------------------
class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=device))
        freqs = 1.0 / freqs
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------
# Tiny U-Net blocks
# ---------------------------
def _group_norm(num_channels: int, num_groups: int = 8):
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.norm2 = _group_norm(out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class TinyUNet(nn.Module):
    """
    Tiny U-Net for MNIST 28x28.
    Predicts eps_theta(x_t, t).
    """
    def __init__(self, in_ch=1, base_ch=64, ch_mult=(1, 2, 4), time_dim=256, dropout=0.0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down blocks
        downs = []
        cur = base_ch
        self.skip_channels = []
        for idx, mult in enumerate(ch_mult):
            out = base_ch * mult
            downs.append(ResBlock(cur, out, time_dim, dropout=dropout))
            downs.append(ResBlock(out, out, time_dim, dropout=dropout))
            self.skip_channels.append(out)
            cur = out
            if idx != len(ch_mult) - 1:
                downs.append(Downsample(cur))
        self.downs = nn.ModuleList(downs)

        # Mid
        self.mid1 = ResBlock(cur, cur, time_dim, dropout=dropout)
        self.mid2 = ResBlock(cur, cur, time_dim, dropout=dropout)

        # Up blocks (reverse)
        ups = []
        for idx, mult in enumerate(reversed(ch_mult)):
            out = base_ch * mult
            ups.append(ResBlock(cur + out, out, time_dim, dropout=dropout))
            ups.append(ResBlock(out, out, time_dim, dropout=dropout))
            cur = out
            if idx != len(ch_mult) - 1:
                ups.append(Upsample(cur))
        self.ups = nn.ModuleList(ups)

        self.out_norm = _group_norm(cur)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(cur, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h = self.init_conv(x)

        skips = []
        i = 0
        # Down path: push skip after each ResBlock pair
        while i < len(self.downs):
            if isinstance(self.downs[i], Downsample):
                h = self.downs[i](h)
                i += 1
                continue
            h = self.downs[i](h, t_emb); i += 1
            h = self.downs[i](h, t_emb); i += 1
            skips.append(h)

        # Mid
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Up path: pop skips
        skips = list(reversed(skips))
        j = 0
        k = 0
        while j < len(self.ups):
            if isinstance(self.ups[j], Upsample):
                h = self.ups[j](h)
                j += 1
                continue
            h = torch.cat([h, skips[k]], dim=1); k += 1
            h = self.ups[j](h, t_emb); j += 1
            h = self.ups[j](h, t_emb); j += 1

        return self.out_conv(self.out_act(self.out_norm(h)))


# ---------------------------
# EMA helper
# ---------------------------
@torch.no_grad()
def update_ema(ema_model, model, decay=0.995):
    """
    EMA update:
      ema = decay * ema + (1-decay) * model
    Tip: 0.995 often works better than 0.999 for shorter trainings.
    """
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------
# Sampling (DDPM with x0 clamp)
# ---------------------------
@torch.no_grad()
def p_sample_ddpm(model, x_t, t, schedules, clamp_x0=True):
    betas_t = extract(schedules["betas"], t, x_t.shape)
    alphas_t = extract(schedules["alphas"], t, x_t.shape)
    abar_t = extract(schedules["alphas_cumprod"], t, x_t.shape)
    abar_prev = extract(schedules["alphas_cumprod_prev"], t, x_t.shape)

    eps_pred = model(x_t, t)
    
    # x0_hat = (x_t - sqrt(1-abar_t)*eps) / sqrt(abar_t)
    x0_hat = (x_t - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t)
    if clamp_x0:
        x0_hat = x0_hat.clamp(-1.0, 1.0)

    # posterior mean: coef1*x0_hat + coef2*x_t
    coef1 = betas_t * torch.sqrt(abar_prev) / (1.0 - abar_t)
    coef2 = (1.0 - abar_prev) * torch.sqrt(alphas_t) / (1.0 - abar_t)
    model_mean = coef1 * x0_hat + coef2 * x_t

    posterior_var_t = extract(schedules["posterior_variance"], t, x_t.shape)

    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (len(x_t.shape) - 1)))
    x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
    return x_prev

@torch.no_grad()
def sample_ddpm(model, schedules, T, image_size=28, num_samples=16, device="cpu", clamp_x0=True):
    model.eval()
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)
    for step in reversed(range(T)):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        x = p_sample_ddpm(model, x, t, schedules, clamp_x0=clamp_x0)
    return x


# ---------------------------
# DDIM sampling (improved)
# ---------------------------
@torch.no_grad()
def sample_ddim(model, schedules, T, steps=200, eta=0.0, image_size=28, num_samples=16,
                device="cpu", clamp_x0=True):
    """
    DDIM sampling with a clean integer timestep schedule.
      - steps: number of sampling steps (try 50, 100, 200)
      - eta: 0.0 deterministic; >0 adds noise
    """
    model.eval()
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)

    # Evenly spaced integer timesteps, then run in reverse.
    step_size = max(T // steps, 1)
    timesteps = list(range(0, T, step_size))
    timesteps[-1] = T - 1  # ensure last is exactly T-1

    for i in reversed(range(len(timesteps))):
        t_val = timesteps[i]
        t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

        abar_t = extract(schedules["alphas_cumprod"], t, x.shape)
        eps = model(x, t)

        # predict x0
        x0_hat = (x - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)
        if clamp_x0:
            x0_hat = x0_hat.clamp(-1.0, 1.0)

        if i == 0:
            x = x0_hat
            break

        t_prev_val = timesteps[i - 1]
        t_prev = torch.full((num_samples,), t_prev_val, device=device, dtype=torch.long)
        abar_prev = extract(schedules["alphas_cumprod"], t_prev, x.shape)

        # sigma controls stochasticity; eta=0 => deterministic
        sigma = eta * torch.sqrt((1.0 - abar_prev) / (1.0 - abar_t)) * torch.sqrt(1.0 - abar_t / abar_prev)
        sigma2 = sigma * sigma

        # Avoid tiny negative due to float error
        coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma2, min=0.0))
        eps_dir = coeff * eps

        noise = torch.randn_like(x)
        x = torch.sqrt(abar_prev) * x0_hat + eps_dir + sigma * noise

    return x


# ---------------------------
# Data + training
# ---------------------------
def to_minus1_plus1(x):
    return x * 2 - 1

def get_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_minus1_plus1),
    ])
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # Windows-safe
        pin_memory=use_cuda
    )

def train(model, ema_model, dataloader, schedules, optimizer, T,
          epochs=120, log_every=200, device="cpu", ema_decay=0.995):
    model.train()
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

            update_ema(ema_model, model, decay=ema_decay)

            running += loss.item()
            if (step + 1) % log_every == 0:
                print(f"Epoch {epoch} Step {step+1}: loss={running/log_every:.4f}")
                running = 0.0

        print(f"Epoch {epoch}: last_batch_loss={loss.item():.4f}")


# ---------------------------
# Visualization + saving
# ---------------------------
def show_samples(samples, nrow=4, title="Samples", save_dir="output_v4", filename=None, show=True):
    os.makedirs(save_dir, exist_ok=True)

    samples = samples.clamp(-1, 1)
    samples = (samples + 1) / 2  # to [0, 1]

    grid = torchvision.utils.make_grid(samples.cpu(), nrow=nrow)

    if filename is None:
        safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in title).strip()
        filename = safe.replace(" ", "_") + ".png"

    save_path = os.path.join(save_dir, filename)
    torchvision.utils.save_image(grid, save_path)

    if show:
        plt.figure(figsize=(4, 4))
        plt.title(title)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()

    print("Saved:", save_path)


# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    T = 1000
    epochs = 120
    batch_size = 128
    lr = 2e-4

    # Separate folder from v2/v3
    save_dir = os.path.join(os.path.dirname(__file__), "output_v4")

    schedules = make_schedules(T, device)

    # Tiny U-Net (works well on MNIST)
    model = TinyUNet(in_ch=1, base_ch=64, ch_mult=(1, 2, 4), time_dim=256, dropout=0.0).to(device)

    # EMA model for sampling
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    dataloader = get_data(batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train(model, ema_model, dataloader, schedules, optimizer, T,
          epochs=epochs, device=device, ema_decay=0.995)

    # ---- Ablation: clamp vs EMA vs DDIM ----
    def set_seed(seed=0):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

    set_seed(0)
    s1 = sample_ddpm(model, schedules, T, num_samples=16, device=device, clamp_x0=False)
    show_samples(s1, title="DDPM_model_no_clamp", save_dir=save_dir, filename="01_ddpm_model_no_clamp.png", show=False)

    set_seed(0)
    s2 = sample_ddpm(model, schedules, T, num_samples=16, device=device, clamp_x0=True)
    show_samples(s2, title="DDPM_model_x0_clamp", save_dir=save_dir, filename="02_ddpm_model_x0_clamp.png", show=False)

    set_seed(0)
    s3 = sample_ddpm(ema_model, schedules, T, num_samples=16, device=device, clamp_x0=True)
    show_samples(s3, title="DDPM_ema_x0_clamp", save_dir=save_dir, filename="03_ddpm_ema_x0_clamp.png", show=False)

    # DDIM: with U-Net this should be closer to DDPM quality
    ddim_steps = 200
    ddim_eta = 0.0  # try 0.2 if still see stroke-y samples

    set_seed(0)
    s4 = sample_ddim(ema_model, schedules, T, steps=ddim_steps, eta=ddim_eta,
                     num_samples=16, device=device, clamp_x0=True)
    show_samples(s4, title=f"DDIM_ema_{ddim_steps}steps_eta{ddim_eta}", save_dir=save_dir,
                 filename="04_ddim_ema_200steps.png", show=False)

    print("Done. Saved all grids to:", save_dir)


if __name__ == "__main__":
    main()
