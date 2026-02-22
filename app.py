

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn


# ==========================================================
# 1️⃣ Load Data
# ==========================================================

DATA_PATH = "data.csv"

if not os.path.exists(DATA_PATH):
    print("Dataset not found.")
    print("Please place 'data.csv' in the project folder.")
    sys.exit()

df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

if "no2" not in df.columns:
    print("Column 'no2' not found in dataset.")
    sys.exit()

x = df["no2"].dropna().values
x = x[(x > 0) & (x < np.percentile(x, 99))]

# Use subset for faster training
x = np.random.choice(x, 20000, replace=False)


# ==========================================================
# 2️⃣ Transformation
# ==========================================================

r = 102303123  # Replace with your roll number

a_r = 0.5 * (r % 7)
b_r = 0.3 * ((r % 5) + 1)

z = x + a_r * np.sin(b_r * x)

# Normalize
z = (z - np.mean(z)) / np.std(z)


# ==========================================================
# 3️⃣ Define WGAN Models
# ==========================================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
# 4️⃣ Training
# ==========================================================

def train_wgan(z, device):

    z_tensor = torch.tensor(z, dtype=torch.float32).view(-1, 1).to(device)

    G = Generator().to(device)
    C = Critic().to(device)

    opt_G = torch.optim.RMSprop(G.parameters(), lr=0.0005)
    opt_C = torch.optim.RMSprop(C.parameters(), lr=0.0005)

    epochs = 2000
    batch_size = 128
    clip_value = 0.01
    critic_steps = 5

    for epoch in range(epochs):

        for _ in range(critic_steps):

            idx = np.random.randint(0, len(z), batch_size)
            real = z_tensor[idx]

            noise = torch.randn(batch_size, 1).to(device)
            fake = G(noise).detach()

            loss_C = -(torch.mean(C(real)) - torch.mean(C(fake)))

            opt_C.zero_grad()
            loss_C.backward()
            opt_C.step()

            for p in C.parameters():
                p.data.clamp_(-clip_value, clip_value)

        noise = torch.randn(batch_size, 1).to(device)
        fake = G(noise)
        loss_G = -torch.mean(C(fake))

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Critic Loss: {loss_C.item():.4f} | G Loss: {loss_G.item():.4f}")

    return G


# ==========================================================
# 5️⃣ Save Results
# ==========================================================

def save_results(z, G, device):

    with torch.no_grad():
        noise = torch.randn(10000, 1).to(device)
        z_fake = G(noise).cpu().numpy().flatten()

    # Real distribution
    plt.hist(z, bins=100, density=True)
    plt.title("Real Distribution")
    plt.savefig("real_distribution.png", dpi=300)
    plt.close()

    # Real vs Generated
    plt.hist(z, bins=100, density=True, alpha=0.5, label="Real")
    plt.hist(z_fake, bins=100, density=True, alpha=0.5, label="Generated")
    plt.legend()
    plt.title("Real vs Generated (WGAN)")
    plt.savefig("wgan_real_vs_generated.png", dpi=300)
    plt.close()

    print("Results saved successfully.")


# ==========================================================
# 6️⃣ Main
# ==========================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    G = train_wgan(z, device)
    save_results(z, G, device)


if __name__ == "__main__":
    main()