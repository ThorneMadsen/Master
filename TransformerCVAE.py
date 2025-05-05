import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, cond_dim, target_dim, latent_dim, d_model=256, nhead=8, num_layers=3):
        super(TransformerEncoder, self).__init__()
        self.input_dim = cond_dim + target_dim
        self.d_model = d_model
        self.embedding = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Linear(self.d_model, latent_dim)
        self.fc_logvar = nn.Linear(self.d_model, latent_dim)
        
    def forward(self, cond, target):
        x = torch.cat([cond, target], dim=2)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Use the output of the last time step instead of the mean
        last_step_output = x[:, -1, :] # Shape: [batch_size, d_model]
        
        mu = self.fc_mu(last_step_output)
        logvar = self.fc_logvar(last_step_output)
        return mu, logvar

class TransformerDecoder(nn.Module):
    def __init__(self, cond_dim, latent_dim, target_dim, d_model=256, nhead=8, num_layers=3, seq_len=24):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.target_dim = target_dim

        # Embedding for the condition sequence to be used as memory
        self.cond_embedding = nn.Linear(cond_dim, self.d_model)
        self.cond_pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len)
        # Using a simple encoder for the condition sequence (could reuse TransformerEncoderLayer)
        memory_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.memory_transformer = nn.TransformerEncoder(memory_encoder_layer, num_layers=num_layers)

        # Projection for latent vector z to match d_model
        self.z_proj = nn.Linear(latent_dim, self.d_model)

        # Embedding/Input layer for the decoder target sequence (starts from z)
        # We create the target sequence input within forward
        self.decoder_pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, # Use d_model
            nhead=nhead,
            dim_feedforward=1024, # Keep or adjust based on d_model
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.target_dim) # Use d_model

    def forward(self, cond_seq, z):
        # cond_seq shape: [batch_size, seq_len, cond_dim]
        # z shape: [batch_size, latent_dim]

        # 1. Embed and positionally encode the condition sequence
        cond_embedded = self.cond_embedding(cond_seq) # [batch_size, seq_len, d_model]
        cond_embedded_pos = self.cond_pos_encoder(cond_embedded)
        
        # 2. Process condition sequence through its own transformer to create context (memory)
        memory = self.memory_transformer(cond_embedded_pos) # [batch_size, seq_len, d_model]

        # 3. Project z and combine it with the memory
        z_projected = self.z_proj(z) # [batch_size, d_model]
        z_expanded = z_projected.unsqueeze(1) # [batch_size, 1, d_model]
        # Add z to each time step's memory vector via broadcasting
        memory_with_z = memory + z_expanded 

        # 4. Prepare the target input sequence for the main decoder
        # Use the initially embedded & positionally encoded conditions as the input sequence `tgt`
        # The decoder's task is to transform this input using the z-infused memory
        tgt_input = cond_embedded_pos 
        # Note: Removed self.decoder_pos_encoder as cond_embedded_pos already has pos encoding

        # 5. Pass through transformer decoder
        # Use the condition-based sequence as `tgt` and the z-infused context as `memory`
        output = self.transformer(tgt=tgt_input, memory=memory_with_z) # [batch_size, seq_len, d_model]

        # 6. Project to target dimension
        out = self.output_proj(output) # [batch_size, seq_len, target_dim]
        return out

class TransformerCVAE(nn.Module):
    def __init__(self, cond_dim, target_dim, latent_dim, d_model=256, nhead=8, num_layers=3, seq_len=24):
        super(TransformerCVAE, self).__init__()
        self.seq_len = seq_len
        self.encoder = TransformerEncoder(cond_dim, target_dim, latent_dim, d_model, nhead, num_layers)
        self.decoder = TransformerDecoder(cond_dim, latent_dim, target_dim, d_model, nhead, num_layers, seq_len)

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
        # cond shape: [batch_size, seq_len, cond_dim]
        # target shape: [batch_size, seq_len, target_dim]
        mu, logvar = self.encoder(cond, target)
        z = self.reparameterize(mu, logvar) # z shape: [batch_size, latent_dim]
        # Pass original condition sequence and latent vector z to decoder
        recon_target = self.decoder(cond, z) # recon_target shape: [batch_size, seq_len, target_dim]
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
        # Ensure inputs are sequences [batch, seq_len, features]
        batch_cond = batch_cond.to(device)       # Expected shape: [batch, 24, 3]
        batch_target = batch_target.to(device)   # Expected shape: [batch, 24, 1]
        # batch_cond_flat = batch_cond.view(batch_cond.size(0), -1) # No longer needed

        optimizer.zero_grad()
        # Pass sequences directly
        recon_target, mu, logvar = model(batch_cond, batch_target)
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
            # Ensure inputs are sequences [batch, seq_len, features]
            batch_cond = batch_cond.to(device)     # Expected shape: [batch, 24, 3]
            batch_target = batch_target.to(device) # Expected shape: [batch, 24, 1]
            # batch_cond_flat = batch_cond.view(batch_cond.size(0), -1) # No longer needed

            # Pass sequences directly
            recon_target, mu, logvar = model(batch_cond, batch_target)
            loss = loss_function(recon_target, batch_target, mu, logvar, kl_weight)
            total_loss += loss.item() * batch_cond.size(0)

    return total_loss / len(val_loader.dataset)

def generate_samples(model, cond, num_samples, device):
    """
    Generate multiple samples for given conditions
    cond shape should be [batch_size, seq_len, cond_dim]
    """
    model.eval()
    samples = []
    with torch.no_grad():
        cond = cond.to(device) # Shape: [batch_size, seq_len, cond_dim]
        # cond_flat = cond.view(cond.size(0), -1) # No longer needed

        batch_size = cond.size(0)

        for _ in range(num_samples):
            # Sample z from prior N(0, I) for the batch
            z = torch.randn(batch_size, model.encoder.fc_mu.out_features).to(device)
            # Generate sample sequence using decoder
            # Decoder now takes the condition sequence and z
            sample = model.decoder(cond, z) # Output shape: [batch_size, seq_len, target_dim]
            samples.append(sample.unsqueeze(0)) # Add sample dimension

    # Concatenate along the sample dimension
    return torch.cat(samples, dim=0)  # [num_samples, batch_size, seq_len, target_dim]

if __name__ == "__main__":
    import os
    import time
    from torch.utils.data import DataLoader, TensorDataset, random_split

    # Use hardcoded settings from notebook instead of argparse
    
    # Model parameters
    # cond_dim is now calculated dynamically from data (see below)
    target_dim = 1        # Number of target features per time step
    latent_dim = 16       # Dimension of latent space
    seq_len = 24          # Sequence length
    d_model = 256         # Model dimension
    nhead = 8             # Number of attention heads
    num_layers = 3        # Number of transformer layers
    
    # Training parameters
    batch_size = 256      # 2^8 as in notebook
    epochs = 100
    lr = 0.001
    warmup_epochs = 10
    save_interval = 10
    
    # Paths
    data_path = "data/X2.npy"
    output_dir = "./results"
    mode = "train"        # 'train' or 'generate'
    model_path = None     # Path to load pretrained model
    num_samples = 10      # Number of samples to generate (for generate mode)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    print(f"Loading data from {data_path}")
    X = np.load(data_path)
    # Update to match MLPVAE indexing:
    target = X[:, 0, :]     # First channel is the target (vs 4th channel in original)
    condition = X[:, 1:, :] # Rest are conditions (vs first 3 in original)

    # Permute changes dimensions from (batch, features, time) to (batch, time, features)
    # This is necessary for transformer which expects sequence data as (batch, sequence_length, features)
    cond_tensor = torch.tensor(condition, dtype=torch.float32).permute(0, 2, 1)
    
    # unsqueeze(-1) adds a feature dimension, converting (batch, time) to (batch, time, 1)
    # This ensures target has shape (batch, time, 1) which the model expects
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
    
    # Update condition dimension to match the actual shape
    cond_dim = condition.shape[1]  # Now dynamically calculated
    
    # Create dataset
    dataset = TensorDataset(cond_tensor, target_tensor)
    
    # Split data
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    if mode == 'train':
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"Data dimensions - Sequence length: {seq_len}, Condition dim: {cond_dim}, Target dim: {target_dim}")
        print(f"Dataset size - Total: {n_total}, Train: {n_train}, Validation: {n_val}")
        
        # Initialize model
        model = TransformerCVAE(
            cond_dim=cond_dim,
            target_dim=target_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len
        ).to(device)
        
        # Load pretrained model if provided
        if model_path:
            print(f"Loading pretrained model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        print(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Calculate KL weight with linear annealing
            kl_weight = calculate_kl_weight(
                epoch=epoch-1, 
                total_epochs=epochs,
                warmup_epochs=warmup_epochs,
                min_weight=0.0,
                max_weight=1.0,
                schedule_type='linear'
            )
            
            # Train and validate
            train_loss = train_epoch(model, train_loader, optimizer, device, kl_weight)
            val_loss = validate(model, val_loader, device, kl_weight)
            
            # Print progress
            print(f"Epoch {epoch}/{epochs} - KL Weight: {kl_weight:.4f} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if epoch % save_interval == 0 or epoch == epochs:
                checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved model checkpoint to {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, "model_final.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Training completed in {time.time() - start_time:.2f}s. Final model saved to {final_model_path}")
        
    elif mode == 'generate':
        # For generation mode, we need a trained model
        if not model_path:
            raise ValueError("Model path must be provided for generate mode")
        
        # Use a subset for testing
        test_dataset, _ = random_split(dataset, [100, len(dataset)-100])  # Just use 100 samples for testing
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get the first batch for generation
        test_cond, _ = next(iter(test_loader))
        
        # Initialize model
        model = TransformerCVAE(
            cond_dim=cond_dim,
            target_dim=target_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            seq_len=seq_len
        ).to(device)
        
        # Load trained model
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Generate samples
        print(f"Generating {num_samples} samples for {len(test_cond)} test conditions")
        samples = generate_samples(model, test_cond, num_samples, device)
        
        # Save generated samples
        samples_path = os.path.join(output_dir, "generated_samples.pt")
        torch.save(samples, samples_path)
        print(f"Saved {num_samples} generated samples to {samples_path}") 