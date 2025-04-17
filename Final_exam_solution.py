# %% [markdown]
# # CS551 Final Exam: Generative Adversarial Neural Networks
#
# **Authors:** Fabio Cozzuto, Johan Mogollon  
# **Contributions:**  
# - **Fabio Cozzuto:** All code, experiments, and analysis  
# - **Johan Mogollon:** All code, experiments, and analysis
#
# This notebook demonstrates:
# 1. Padding calculation for DCGAN discriminator  
# 2. Data‑augmentation pipelines (`basic` vs. `deluxe`)  
# 3. Visualizing DCGAN samples  
# 4. Plotting DCGAN training losses  
# 5. Comparing CycleGAN outputs

# %% [markdown]
# ## 1. Environment Setup & Imports

import os, sys
# Ensure Jupyter can import our GAN modules
sys.path.insert(0, os.path.abspath('.'))

import torch
from models import DCGenerator, DCDiscriminator, CycleGenerator
from utils import to_var, to_data
from data_loader import get_data_loader
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# %% [markdown]
# ## 2. Padding Calculation for DCGAN Discriminator
#
# **Question:** With kernel size \(K=4\) and stride \(S=2\), what padding \(P\) halves the spatial dimensions?  
#
# **Answer:** The convolution output formula is  
# \[
#   \text{out} = \frac{\text{in} - K + 2P}{S} + 1.
# \]
# Setting \(\text{out} = \tfrac{\text{in}}{2}\) and solving gives \(P = 1\). :contentReference[oaicite:2]{index=2}

# %% [markdown]
# ## 3. Data‑Augmentation Pipelines
#
# We define both **basic** and **deluxe** transforms.  
# Deluxe uses a 10% up‑scale + random crop + flip :contentReference[oaicite:3]{index=3}.

# %%
from data_loader import get_data_loader  # re-import for clarity

# Suppose opts.image_size is 64 for this exam
image_size = 64

basic_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

deluxe_transform = transforms.Compose([
    transforms.Resize(int(1.1 * image_size)),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

# Display example
sample_img = Image.open('data/sample.png').convert('RGB')
fig, axes = plt.subplots(1, 2, figsize=(8,4))
axes[0].imshow(np.transpose(basic_transform(sample_img).numpy(), (1,2,0)))
axes[0].set_title("Basic")
axes[0].axis('off')
axes[1].imshow(np.transpose(deluxe_transform(sample_img).numpy(), (1,2,0)))
axes[1].set_title("Deluxe")
axes[1].axis('off')
plt.show()

# %% [markdown]
# ## 4. Visualizing DCGAN Samples
#
# We use our `DCGenerator` and display a 4×4 grid of generated images :contentReference[oaicite:4]{index=4}.

# %%
# Instantiate generator
G = DCGenerator(noise_size=100, conv_dim=64).cuda()
fixed_noise = to_var(torch.randn(16, 100, 1, 1))

# Generate samples
with torch.no_grad():
    fake_images = G(fixed_noise)

# Make grid and plot
grid = make_grid(fake_images, nrow=4, normalize=True, value_range=(-1,1))
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1,2,0))
plt.title("DCGAN Fake Samples")
plt.axis('off')
plt.show()

# %% [markdown]
# ## 5. DCGAN Training Loss Curves
#
# Load your logged losses (saved as NumPy arrays during training).

# %%
# Replace with your actual log paths
g_losses = np.load('logs/dcgan_g_losses.npy')
d_losses = np.load('logs/dcgan_d_losses.npy')
iterations = np.arange(len(g_losses)) * 100  # e.g. logged every 100 iters

plt.figure(figsize=(8,4))
plt.plot(iterations, g_losses, label='Generator Loss')
plt.plot(iterations, d_losses, label='Discriminator Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("DCGAN Training Curves")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. CycleGAN Sample Comparisons
#
# Display saved sample images from 𝑋→𝑌 and 𝑌→𝑋 at iteration 400.

# %%
def show_image(path, title):
    img = Image.open(path).convert('RGB')
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

show_image('output/cyclegan/sample-000400-X-Y.png', 'CycleGAN X→Y @400')
show_image('output/cyclegan/sample-000400-Y-X.png', 'CycleGAN Y→X @400')

# %% [markdown]
# ## 7. Cycle Consistency Loss
#
# The cycle loss enforces \(G(F(y)) \approx y\) and \(F(G(x)) \approx x\), typically using L1:
#
# \[
#   \mathcal{L}_{cycle}
#   = \mathbb{E}_{x\sim X}\lVert F(G(x)) - x\rVert_1
#   + \mathbb{E}_{y\sim Y}\lVert G(F(y)) - y\rVert_1.
# \]

# %% [markdown]
# ## 8. Embedding TensorBoard in‑Notebook
#
# Launch TensorBoard directly in this notebook :contentReference[oaicite:5]{index=5}.

# %%
# In a Jupyter cell, uncomment to launch:
# %load_ext tensorboard
# %tensorboard --logdir=output/vanilla  # or your CycleGAN logs