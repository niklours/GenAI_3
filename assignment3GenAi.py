# %%
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%


# ----------------------------
# Encoder Network
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ----------------------------
# Decoder Network (Learned Variance)
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv_base = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU()
        )

        # Two output heads:
        self.out_mu = nn.Conv2d(32, 1, kernel_size=3, padding=1)       # Mean image
        self.out_logvar = nn.Conv2d(32, 1, kernel_size=3, padding=1)   # Log-variance

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        x = self.deconv_base(x)
        mu = torch.sigmoid(self.out_mu(x))  # constrain to [0,1] pixel space
        logvar = self.out_logvar(x)         # unconstrained
        logvar = torch.clamp(logvar, min=-6.0, max=3.0)  # Clamp safely here
        return mu, logvar

# ----------------------------
# VAE Wrapper
# ----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)                 # encoder outputs
        z = self.reparameterize(mu_z, logvar_z)          # latent sample
        mu_x, logvar_x = self.decoder(z)                 # decoder outputs
        return mu_x, logvar_x, mu_z, logvar_z, z         # outputs needed for ELBO

# %%


def get_mnist_dataloaders(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_set, val_set = random_split(full_train_set, [50000, 10000])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

def elbo_loss(x, mu_x, logvar_x, mu_z, logvar_z):
    B = x.size(0)
    x = x.view(B, -1)
    mu_x = mu_x.view(B, -1)
    logvar_x = logvar_x.view(B, -1)

    eps = 1e-6
    var = torch.exp(logvar_x) + eps  # Safe variance

    # Numerically stable Gaussian log-likelihood
    recon_loss = 0.5 * (
        torch.log(2 * torch.pi * var) + ((x - mu_x) ** 2) / var
    )
    recon_loss = torch.sum(recon_loss, dim=1)

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)

    return torch.mean(recon_loss + kl_div)

# %%

def train_vae(model, train_loader, val_loader, optimizer,
                                   num_epochs=100, patience=10, device='cuda'):

    model = model.to(device)
    train_elbo, val_elbo = [], []

    best_val_elbo = float('-inf')
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            mu_x, logvar_x, mu_z, logvar_z, _ = model(x_batch)
            loss = elbo_loss(x_batch, mu_x, logvar_x, mu_z, logvar_z)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_elbo.append(-avg_train_loss)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                mu_x, logvar_x, mu_z, logvar_z, _ = model(x_val)
                loss = elbo_loss(x_val, mu_x, logvar_x, mu_z, logvar_z)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_elbo.append(-avg_val_loss)

        print(f"Epoch {epoch:02d} | Train ELBO: {-avg_train_loss:.4f} | Val ELBO: {-avg_val_loss:.4f}")

        # Early stopping logic
        if -avg_val_loss > best_val_elbo:
            best_val_elbo = -avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Stopping early at epoch {epoch} due to no improvement in validation ELBO.")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Plot ELBO
    plt.plot(train_elbo, label='Train ELBO')
    plt.plot(val_elbo, label='Validation ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.legend()
    plt.title('ELBO vs Epoch')
    plt.show()

    return train_elbo, val_elbo



# %%

def test_vae_reconstruction_and_generation(model, test_loader, device='cuda'):
    model.eval()
    model = model.to(device)

    # Get a single batch from test set
    x_test, _ = next(iter(test_loader))
    x_test = x_test[:32].to(device)  # use only first 32 for display
    with torch.no_grad():
        mu_z, logvar_z = model.encoder(x_test)
        z = model.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = model.decoder(z)
        std_x = torch.exp(0.5 * logvar_x)
        recon_x = mu_x + std_x * torch.randn_like(std_x)  # sample x' ~ N(mu, sigma²)

    # Arrange originals and reconstructions side-by-side
    grid = torch.cat([x_test.cpu(), recon_x.cpu()], dim=0)
    grid_img = vutils.make_grid(grid, nrow=8, pad_value=1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Top: Original | Bottom: Reconstruction')
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()

def generate_from_prior(model, num_samples=64, device='cuda'):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        z = torch.randn(num_samples, model.encoder.fc_mu.out_features).to(device)
        mu_x, logvar_x = model.decoder(z)
        std_x = torch.exp(0.5 * logvar_x)
        gen_x = mu_x + std_x * torch.randn_like(std_x)

    grid_img = vutils.make_grid(gen_x.cpu(), nrow=8, pad_value=1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Generated Samples from Prior')
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()


# %%
#Run these commands for Task 1
# torch.manual_seed(42)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# latent_dim = 20
# batch_size = 128

# train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size)

# model = VAE(latent_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# model_20=model


# train_vae(model, train_loader, val_loader, optimizer, num_epochs=100, patience=10, device=device)
# test_vae_reconstruction_and_generation(model, test_loader, device=device)
# generate_from_prior(model, num_samples=64, device=device)

# %% [markdown]
# Save this model for tasks 3 and 4

# %%


# %% [markdown]
# Task 2

# %%


# ----------------------------
# Encoder Network
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 14 -> 7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7 (added)
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# ----------------------------
# Decoder Network (Using Beta)
# ----------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv_base = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.alpha_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.beta_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 7, 7)
        x = self.deconv_base(x)
        alpha = F.softplus(self.alpha_head(x)) + 1e-3  
        beta = F.softplus(self.beta_head(x)) + 1e-3
        return alpha, beta

# ----------------------------
# VAE Wrapper
# ----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        alpha, beta = self.decoder(z)
        return alpha, beta, mu_z, logvar_z, z

# %%


def get_mnist_dataloaders(batch_size=128):
    def transform_fn(x):
        x = transforms.ToTensor()(x)
        return (x * 0.98 + 0.01)  # now in [0.01, 0.99]
    
    full_train_set = datasets.MNIST('./data', train=True, download=True, transform=transform_fn)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform_fn)
    train_set, val_set = random_split(full_train_set, [50000, 10000])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False)
    )

def elbo_loss(x, alpha, beta, mu_z, logvar_z, beta_kl):
    B = x.size(0)
    x = x.view(B, -1)
    alpha = alpha.view(B, -1)
    beta = beta.view(B, -1)

    eps = 1e-6
    x = torch.clamp(x, eps, 1 - eps)

    log_likelihood = (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1) * torch.log(x)
        + (beta - 1) * torch.log(1 - x)
    )
    recon_loss = -torch.sum(log_likelihood, dim=1)

    kl_div = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)

    return torch.mean(recon_loss + beta_kl * kl_div)

# %%

def train_vae(model, train_loader, val_loader, optimizer,
                                   num_epochs=100, patience=10, device='cuda'):

    model = model.to(device)
    train_elbo, val_elbo = [], []

    best_val_elbo = float('-inf')
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        #beta_kl = min(10.0, epoch / 50 * 10)
        beta_kl = 10

        model.train()
        total_train_loss = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            mu_x, logvar_x, mu_z, logvar_z, _ = model(x_batch)
            loss = elbo_loss(x_batch, mu_x, logvar_x, mu_z, logvar_z, beta_kl)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_elbo.append(-avg_train_loss)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device)
                mu_x, logvar_x, mu_z, logvar_z, _ = model(x_val)
                loss = elbo_loss(x_val, mu_x, logvar_x, mu_z, logvar_z, beta_kl)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_elbo.append(-avg_val_loss)
        

        print(f"Epoch {epoch:02d} | Train ELBO: {-avg_train_loss:.4f} | Val ELBO: {-avg_val_loss:.4f}")

        # Early stopping logic
        if -avg_val_loss > best_val_elbo:
            best_val_elbo = -avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Stopping early at epoch {epoch} due to no improvement in validation ELBO.")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    # Plot ELBO
    plt.plot(train_elbo, label='Train ELBO')
    plt.plot(val_elbo, label='Validation ELBO')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.legend()
    plt.title('ELBO vs Epoch')
    plt.show()

    return train_elbo, val_elbo

# %%

def test_vae_reconstruction_and_generation(model, test_loader, device='cuda'):
    model.eval()
    model = model.to(device)

    # Get a single batch from test set
    x_test, _ = next(iter(test_loader))
    x_test = x_test[:32].to(device)  # use only first 32 for display
    with torch.no_grad():
        mu_z, logvar_z = model.encoder(x_test)
        z = model.reparameterize(mu_z, logvar_z)
        alpha, beta = model.decoder(z)
        alpha = torch.clamp(alpha, min=1e-3)
        beta = torch.clamp(beta, min=1e-3)
        recon_x = torch.distributions.Beta(alpha, beta).sample()

    # Arrange originals and reconstructions side-by-side
    grid = torch.cat([x_test.cpu(), recon_x.cpu()], dim=0)
    grid_img = vutils.make_grid(grid, nrow=8, pad_value=1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Top: Original | Bottom: Reconstruction')
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()

def generate_from_prior(model, num_samples=64, device='cuda'):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        z = torch.randn(num_samples, model.encoder.fc_mu.out_features).to(device)
        alpha, beta = model.decoder(z)
        # alpha = torch.clamp(alpha, min=1e-3)
        # beta = torch.clamp(beta, min=1e-3)
        gen_x = torch.distributions.Beta(alpha, beta).sample()


    grid_img = vutils.make_grid(gen_x.cpu(), nrow=8, pad_value=1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Generated Samples from Prior')
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()

# %%
#Run these commands for Task 2

# torch.manual_seed(42)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# latent_dim = 25
# batch_size = 128

# train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size)

# model = VAE(latent_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Run these commands for Task 2

# train_vae(model, train_loader, val_loader, optimizer, num_epochs=300, patience=10, device=device)
# test_vae_reconstruction_and_generation(model, test_loader, device=device)
# generate_from_prior(model, num_samples=64, device=device)

# %% [markdown]
# TASK 3

# %%
#Run these commands for task 3

# torch.manual_seed(42)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# latent_dim = 2
# batch_size = 128

# train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size)

# model = VAE(latent_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# train_elbo, val_elbo = train_vae(model, train_loader, val_loader, optimizer, device='cuda')

# %%
#model.load_state_dict(torch.load("vae_latent_2d.pth"))

def get_latents(model, test_loader, device, threshold = 1000):
    latent_var = []
    labels = []
    count = 0

    model = model.to(device)
    model.eval()
    
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        mu_z, _ = model.encoder(x_batch)
        latent_var.append(mu_z.detach().cpu().numpy())
        labels.append(y_batch.cpu().numpy())
        count += int(x_batch.size(0))
        if threshold <= count:
            break

    latents = np.vstack(latent_var)[:threshold]
    labels = np.hstack(labels)[:threshold]
    return latents, labels
    

def visualize(latents, labels, title_name, x_name, y_name):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c = labels, cmap = 'tab10', s = 12)
    plt.colorbar(scatter, ticks=range(10), label='Digit class')
    plt.title(title_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True)
    plt.show()


#Run these commands for Task 3a

# latents,labels = get_latents(model, test_loader, device='cuda', threshold=1000)
# visualize(latents, labels, "Latent Dimention", "Latent dimention 1", "Latent dimention 2")

# %%
#Run these commands for task 3

# latent_dim = 10
# train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size)

# model_10 = VAE(latent_dim)
# optimizer = torch.optim.Adam(model_10.parameters(), lr=1e-4)

# train_elbo, val_elbo = train_vae(model_10, train_loader, val_loader, optimizer, device='cuda')

# %%
# U, labels = get_latents(model_10, test_loader, device=device, threshold=1000)
# print(f"Shape of matrix U: {U.shape}")

## PCA analysis on matrix U
# pca = PCA(n_components=2)
# U_pca = pca.fit_transform(U)

#Run these commands for Task 3b
# visualize(U_pca, labels, "PCA Latent Dimention", "pca 1", "pca 2")

# %%

def random_latent_sample(model, x, device='cuda'):
    model.eval()
    model = model.to(device)
    x = x.to(device)
    with torch.no_grad():
        mu_z, logvar_z = model.encoder(x.unsqueeze(0))
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z
    return z.squeeze(0)

def decode_latent(model, z, device='cuda'):
    model.eval()
    with torch.no_grad():
        mu_x, logvar_x = model.decoder(z.unsqueeze(0))
    return mu_x.squeeze(0).cpu()

# %% [markdown]
# Task 3c

# %%


def compute_interpolation(model, test_loader, num_rows=5, num_steps=6, device='cuda'):
    lamdas = np.linspace(0, 1, num_steps)
    dataset = test_loader.dataset
    all_rows = []

    for _ in range(num_rows):
        while True:
            idx1 = random.randint(0, len(dataset)-1)
            idx2 = random.randint(0, len(dataset)-1)
            x1, y1 = dataset[idx1]
            x2, y2 = dataset[idx2]
            if y1 != y2:
                break

        z1 = random_latent_sample(model, x1, device=device)
        z2 = random_latent_sample(model, x2, device=device)

        row_images = []
        row_images.append(x1)  # initial image

        for lamda in lamdas:
            linear_interpolation = lamda * z1 + (1 - lamda) * z2
            decode_z = decode_latent(model, linear_interpolation, device=device)
            row_images.append(decode_z)

        row_images.append(x2)  # target image

        row_tensor = torch.stack([
            img if isinstance(img, torch.Tensor) else img.data for img in row_images
        ])
        all_rows.append(row_tensor)

    grid = torch.cat(all_rows, dim=0)
    grid_img = vutils.make_grid(grid, nrow=num_steps+2, pad_value=1)

    plt.figure(figsize=(num_steps+2, num_rows * 2))
    plt.axis('off')
    plt.title("Full Interpolation Grid (Task 3c)")
    plt.imshow(grid_img.permute(1, 2, 0), cmap='gray')
    plt.show()
#Run these commands for Task 3c

#compute_interpolation(model_20, test_loader,num_rows=5, num_steps=6, device=device)    

    
    
        

# %% [markdown]
# Task 4

# %%
class InferenceOptimizer(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(latent_dim))
        self.logvar = nn.Parameter(torch.zeros(latent_dim))

    def sample_z(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps * std

# %%
def elbo_single_point(x, decoder, infer_module, num_samples=5):
    elbo = 0
    for _ in range(num_samples):
        z = infer_module.sample_z()
        mu_x, logvar_x = decoder(z.unsqueeze(0))
        logvar_x = torch.clamp(logvar_x, min=-6.0, max=3.0)
        recon_loss = 0.5 * (logvar_x + ((x - mu_x) ** 2) / torch.exp(logvar_x)).sum()
        kl = -0.5 * torch.sum(1 + infer_module.logvar - infer_module.mu.pow(2) - infer_module.logvar.exp())
        elbo += recon_loss + kl
    return elbo / num_samples


# %%
def optimize_inference(x, decoder, latent_dim, lr=1e-2, max_steps=2000, patience=100, tol=1e-4, num_samples=1):
    infer = InferenceOptimizer(latent_dim).to(x.device)
    optimizer = torch.optim.Adam(infer.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        loss = elbo_single_point(x, decoder, infer, num_samples=num_samples)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - tol:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return infer



# %% [markdown]
# We use an early stopping criterion based on validation of the ELBO loss. At each optimization step, we track the best ELBO obtained so far. If no significant improvement (greater than 1e-4) is observed for 100 consecutive steps, we terminate the optimization. We also impose a hard cap of 2000 steps.

# %% [markdown]
# We also experimented with using multiple Monte Carlo samples to estimate the expectation. This resulted in a huge increase in model performance.
# 
# 

# %%
def compare_reconstructions(model, test_loader, latent_dim, device='cuda',num_samples=1):
    model.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:5].to(device)

    fig, axes = plt.subplots(len(images), 3, figsize=(9, len(images) * 2))
    for i, x in enumerate(images):
        infer = optimize_inference(x, model.decoder, latent_dim,num_samples=num_samples)
        z_opt = infer.sample_z().unsqueeze(0)
        mu_x_opt, logvar_x_opt = model.decoder(z_opt)
        std_x_opt = torch.exp(0.5 * logvar_x_opt)
        recon_opt = mu_x_opt + std_x_opt * torch.randn_like(std_x_opt)

        with torch.no_grad():
            mu_z, logvar_z = model.encoder(x.unsqueeze(0))
            z = model.reparameterize(mu_z, logvar_z)
            mu_x_std, logvar_x_std = model.decoder(z)
            std_x_std = torch.exp(0.5 * logvar_x_std)
            recon_std = mu_x_std + std_x_std * torch.randn_like(std_x_std)

        axes[i, 0].imshow(x.squeeze().cpu(), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 1].imshow(recon_opt.squeeze().detach().cpu(), cmap='gray')
        axes[i, 1].set_title('Optimized')
        axes[i, 2].imshow(recon_std.squeeze().detach().cpu(), cmap='gray')
        axes[i, 2].set_title('Encoder')

        for ax in axes[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# %%
#Run these commands for task 4a
#compare_reconstructions(model_20, test_loader, latent_dim=20, device=device)

# %% [markdown]
# Using multiple samples for the Monte Carlo estimate significantly improves the quality of the generated reconstructions.

# %%
#Run these commands for Task 4a with 5 sample MC
#compare_reconstructions(model_20, test_loader, latent_dim=20, device=device,num_samples=5)

# %%
def elbo_left_half(x_left, decoder, infer_module, num_samples=1):
    elbo = 0
    for _ in range(num_samples):
        z = infer_module.sample_z()
        mu_x, logvar_x = decoder(z.unsqueeze(0))  
        logvar_x = torch.clamp(logvar_x, min=-6.0, max=3.0)
        mu_left = mu_x[:, :, :, :14]
        logvar_left = logvar_x[:, :, :, :14]

        recon_loss = 0.5 * (logvar_left + ((x_left.unsqueeze(0) - mu_left) ** 2) / torch.exp(logvar_left)).sum()
        kl = -0.5 * torch.sum(1 + infer_module.logvar - infer_module.mu.pow(2) - infer_module.logvar.exp())
        elbo += recon_loss + kl

    return elbo / num_samples



# %%
def optimize_inference_left(x, decoder, latent_dim, lr=1e-2, max_steps=2000, patience=100, tol=1e-4):
    infer = InferenceOptimizer(latent_dim).to(x.device)
    optimizer = torch.optim.Adam(infer.parameters(), lr=lr)

    x_left = x[:, :, :, :14]  
    best_loss = float("inf")
    patience_counter = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        loss = elbo_left_half(x_left, decoder, infer, num_samples=5)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - tol:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return infer



# %% [markdown]
# We use an early stopping criterion based on validation of the ELBO loss. At each optimization step, we track the best ELBO obtained so far. If no significant improvement (greater than 1e-4) is observed for 100 consecutive steps, we terminate the optimization. We also impose a hard cap of 2000 steps.

# %%
def complete_images(model, test_loader, latent_dim, device='cuda'):
    model.eval()
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    images = images[:5].to(device)

    fig, axes = plt.subplots(len(images), 3, figsize=(9, len(images) * 2))

    for i, x in enumerate(images):
        x = x.unsqueeze(0)  

        infer = optimize_inference_left(x, model.decoder, latent_dim)
        z_opt = infer.sample_z().unsqueeze(0)
        mu_x, logvar_x = model.decoder(z_opt)
        std_x = torch.exp(0.5 * logvar_x)
        completed = mu_x + std_x * torch.randn_like(std_x)

        half_masked = torch.zeros_like(x)
        half_masked[:, :, :, :14] = x[:, :, :, :14]

        completed_combined = x.clone()
        completed_combined[:, :, :, 14:] = completed[:, :, :, 14:]

        axes[i, 0].imshow(x.squeeze().detach().cpu(), cmap='gray')
        axes[i, 0].set_title("Original")

        axes[i, 1].imshow(half_masked.squeeze().detach().cpu(), cmap='gray')
        axes[i, 1].set_title("Masked Left Only")

        axes[i, 2].imshow(completed_combined.squeeze().detach().cpu(), cmap='gray')
        axes[i, 2].set_title("Completed")

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# %%
#Run these commands for Task 4b

#complete_images(model, test_loader, latent_dim=20, device=device)


# %%



