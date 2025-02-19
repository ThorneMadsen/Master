import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Ensure data is processed
if not os.path.exists("data/X.npy") or not os.path.exists("data/Y.npy"):
    print("Processed data not found. Running VAE_dataprocessing.py...")
    os.system("python VAE_dataprocessing.py")

# Load processed training data
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Create a dataset and dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CVAE model
class CVAE(nn.Module):
    def __init__(self, input_dim=24, latent_dim=64):
        super(CVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(64 * 24, latent_dim)
        self.fc_log_var = nn.Linear(64 * 24, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 1536)  # Ensure correct expansion
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (64, 24)),  # Correctly match 1536 output
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10))  # Clip log_var
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = self.fc_mean(encoded), self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder_fc(z).view(-1, 64, 24)  # Ensure correct reshaping
        return self.decoder(decoded), mu, log_var

# Initialize the model, optimizer, and loss function
model = CVAE()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
loss_function = nn.L1Loss()

# Training loop
def train(model, dataloader, epochs=100):
    model.train()
    kl_weight = 0.5  # Strengthen KL loss
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            reconstructed, mu, log_var = model(batch_X)
            
            # Reconstruction Loss (MAE)
            recon_loss = loss_function(reconstructed.squeeze(), batch_Y)

            # KL Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # Total Loss
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Run training
if __name__ == "__main__":
    train(model, dataloader)
