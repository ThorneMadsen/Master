import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, cond_dim, target_dim, latent_dim, nhead=8, num_layers=3):
        super(TransformerEncoder, self).__init__()
        self.input_dim = cond_dim + target_dim
        self.embedding = nn.Linear(self.input_dim, 256)  # Embed to higher dimension
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
    def forward(self, cond, target):
        x = torch.cat([cond, target], dim=1)
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class TransformerDecoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, target_dim, nhead=8, num_layers=3):
        super(TransformerDecoder, self).__init__()
        self.input_dim = cond_dim + latent_dim
        self.embedding = nn.Linear(self.input_dim, 256)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(256, target_dim)
        
    def forward(self, cond, z):
        x = torch.cat([cond, z], dim=1)
        x = self.embedding(x).unsqueeze(1)  # Add sequence dimension
        
        # Create memory for decoder (can be enhanced with more sophisticated memory)
        memory = x
        
        # Pass through transformer decoder
        x = self.transformer(x, memory)
        x = x.squeeze(1)  # Remove sequence dimension
        
        out = self.output_proj(x)
        return out

class TransformerCVAE(nn.Module):
    def __init__(self, cond_dim, target_dim, latent_dim, nhead=8, num_layers=3):
        super(TransformerCVAE, self).__init__()
        self.encoder = TransformerEncoder(cond_dim, target_dim, latent_dim, nhead, num_layers)
        self.decoder = TransformerDecoder(cond_dim, latent_dim, target_dim, nhead, num_layers)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the latent Gaussian
        :param logvar: Log variance of the latent Gaussian
        :return: Sampled latent vector
        """
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, cond, target):
        mu, logvar = self.encoder(cond, target)
        z = self.reparameterize(mu, logvar)
        recon_target = self.decoder(cond, z)
        return recon_target, mu, logvar

def loss_function(recon_target, target, mu, logvar, kl_weight=1.0):
    """
    Computes the VAE loss function with KL weight for better training control.
    recon_target: reconstructed target
    target: input target
    mu: mean of the latent Gaussian
    logvar: log-variance of the latent Gaussian
    kl_weight: weight of the KL divergence term
    """
    recon_loss = F.mse_loss(recon_target, target, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss

def calculate_kl_weight(epoch, total_epochs, warmup_epochs, min_weight=0.0, max_weight=1.0, schedule_type='linear'):
    """
    Calculate the KL weight based on annealing schedule.
    
    Args:
        epoch (int): Current epoch
        total_epochs (int): Total number of epochs for training
        warmup_epochs (int): Number of epochs to reach max_weight
        min_weight (float): Minimum KL weight at the beginning
        max_weight (float): Maximum KL weight after warmup
        schedule_type (str): Type of schedule ('linear', 'sigmoid', or 'cyclical')
        
    Returns:
        float: Current KL weight
    """
    if schedule_type == 'linear':
        # Linear annealing from min_weight to max_weight
        return min_weight + (max_weight - min_weight) * min(1.0, epoch / warmup_epochs)
    
    elif schedule_type == 'sigmoid':
        # Sigmoid annealing for smoother transition
        if warmup_epochs > 0:
            ratio = epoch / warmup_epochs
            return min_weight + (max_weight - min_weight) * (1 / (1 + np.exp(-10 * (ratio - 0.5))))
        else:
            return max_weight
    
    elif schedule_type == 'cyclical':
        # Cyclical annealing with a cycle length of 2*warmup_epochs
        if warmup_epochs > 0:
            cycle_length = 2 * warmup_epochs
            cycle = (epoch % cycle_length) / cycle_length
            if cycle < 0.5:
                # Increasing part of the cycle
                return min_weight + (max_weight - min_weight) * (2 * cycle)
            else:
                # Keep at max for the second half of the cycle
                return max_weight
        else:
            return max_weight
    
    else:
        # Default to constant max weight
        return max_weight

def train_epoch(model, train_loader, optimizer, device, kl_weight=1.0):
    """
    Training loop for one epoch
    """
    model.train()
    total_loss = 0
    for batch_cond, batch_target in train_loader:
        batch_cond = batch_cond.to(device)
        batch_target = batch_target.to(device)
        batch_cond_flat = batch_cond.view(batch_cond.size(0), -1)
        
        optimizer.zero_grad()
        recon_target, mu, logvar = model(batch_cond_flat, batch_target)
        loss = loss_function(recon_target, batch_target, mu, logvar, kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_cond.size(0)
    
    return total_loss / len(train_loader.dataset)

def validate(model, val_loader, device, kl_weight=1.0):
    """
    Validation loop
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_cond, batch_target in val_loader:
            batch_cond = batch_cond.to(device)
            batch_target = batch_target.to(device)
            batch_cond_flat = batch_cond.view(batch_cond.size(0), -1)
            
            recon_target, mu, logvar = model(batch_cond_flat, batch_target)
            loss = loss_function(recon_target, batch_target, mu, logvar, kl_weight)
            total_loss += loss.item() * batch_cond.size(0)
    
    return total_loss / len(val_loader.dataset)

def generate_samples(model, cond, num_samples, device):
    """
    Generate multiple samples for given conditions
    """
    model.eval()
    samples = []
    with torch.no_grad():
        cond = cond.to(device)
        cond_flat = cond.view(cond.size(0), -1)
        
        for _ in range(num_samples):
            # Sample from standard normal distribution
            z = torch.randn(cond.size(0), model.decoder.input_dim - cond_flat.size(1)).to(device)
            # Generate sample using decoder
            sample = model.decoder(cond_flat, z)
            samples.append(sample.unsqueeze(0))
    
    return torch.cat(samples, dim=0)  # [num_samples, batch_size, target_dim] 