import torch
import numpy as np
import os
import matplotlib.pyplot as plt # Added import back
from TransformerCVAE import TransformerCVAE, generate_samples
from torch.utils.data import DataLoader, TensorDataset, random_split

def pinball_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    loss_tensor = torch.where(error >= 0, quantile * error, (quantile - 1) * error)
    return torch.mean(loss_tensor)

def calculate_pinball_losses(y_true, y_samples, quantiles=[0.1, 0.5, 0.9]):
    results = {}
    for q in quantiles:
        # Calculate q-th quantile along sample dimension
        y_q = torch.quantile(y_samples, q, dim=0)
        # Calculate pinball loss
        loss_q = pinball_loss(y_true, y_q, q)
        results[f'pinball_{q:.2f}'] = loss_q.item()
    
    return results

def calculate_crps(y_true, y_samples):
    # Reshape y_true and y_samples for consistent calculation
    batch_size, seq_len, target_dim = y_true.shape
    num_samples = y_samples.shape[0]
    
    # Reshape to match the required calculation format
    # We'll treat each time step as an individual sample for CRPS calculation
    y_true_flat = y_true.reshape(-1, target_dim)  # [batch_size*seq_len, target_dim]
    y_samples_flat = y_samples.permute(0, 2, 1, 3)  # [num_samples, seq_len, batch_size, target_dim]
    y_samples_flat = y_samples_flat.reshape(num_samples, -1, target_dim)  # [num_samples, batch_size*seq_len, target_dim]
    
    # Now calculate CRPS using the same approach as in MLPVAE
    # Expand y_true to match y_samples shape for broadcasting
    y_true_expanded = y_true_flat.unsqueeze(0).expand(num_samples, *y_true_flat.shape)
    
    # First term: average absolute difference between each sample and the true value
    term1 = torch.abs(y_samples_flat - y_true_expanded).mean(dim=0)
    
    # Second term: average pairwise absolute differences among samples
    # Compute pairwise differences along the sample axis
    diff = torch.abs(y_samples_flat.unsqueeze(0) - y_samples_flat.unsqueeze(1))
    term2 = diff.mean(dim=(0, 1))
    
    # CRPS per instance and target dimension
    crps = term1 - 0.5 * term2
    
    # Return the average CRPS over all instances and target dimensions
    return crps.mean().item()

def evaluate_imbalance_improvement(y_true, y_samples, percentile=95):
    """
    Evaluate how often the model's predictions are better (closer to 0) than actual imbalances
    Using the specified percentile of samples as a conservative estimate
    """
    # Get percentile of absolute values for each time point
    abs_samples = torch.abs(y_samples)  # [num_samples, batch, seq_len, 1]
    conservative_pred = torch.quantile(abs_samples, percentile/100, dim=0)  # [batch, seq_len, 1]
    
    # Compare with absolute actual values
    actual_abs = torch.abs(y_true)  # [batch, seq_len, 1]
    
    # Count where our conservative estimate is better (closer to 0)
    improvements = (conservative_pred < actual_abs).sum().item()
    total_points = y_true.numel()
    
    # Calculate average absolute difference
    abs_diff = (conservative_pred - actual_abs).abs().mean().item()
    
    # Calculate relative improvement, handling zero/near-zero values
    epsilon = 1e-5  # threshold for considering a value significantly non-zero
    mask = actual_abs > epsilon
    if mask.sum() > 0:
        # Calculate relative change in imbalance
        # If prediction is better (smaller), this will be positive
        # If prediction is worse (larger), this will be negative
        relative_changes = (actual_abs[mask] - conservative_pred[mask]) / actual_abs[mask]
        # Clip extreme values to reasonable range (-1 means doubled, +1 means reduced to 0)
        relative_changes = torch.clamp(relative_changes, min=-1.0, max=1.0)
        relative_improvement = relative_changes.mean().item() * 100
    else:
        relative_improvement = 0.0
    
    return {
        'improvement_percentage': improvements / total_points * 100,
        'avg_absolute_diff': abs_diff,
        'relative_improvement': relative_improvement
    }

def main():
    # Configuration
    model_path = "./results/model_final.pt"
    data_path = "data/X2.npy"
    output_dir = "./evaluation_results"
    num_samples = 100
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {data_path}")
    X = np.load(data_path)
    
    # Update to match MLPVAE indexing:
    target = X[:, 0, :]     # First channel is the target (vs 4th channel in original)
    condition = X[:, 1:, :] # Rest are conditions (vs first 3 in original)

    # Permute changes dimensions from (batch, features, time) to (batch, time, features)
    cond_tensor = torch.tensor(condition, dtype=torch.float32).permute(0, 2, 1)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
    
    # Create dataset
    dataset = TensorDataset(cond_tensor, target_tensor)
    
    # Use the same validation split as in training (last 20%)
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_total))
    
    # Create dataloader with batch_size=1 to process one day at a time
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"\nEvaluating validation set ({n_val} days) with {num_samples} samples per day")
    
    # Get model parameters
    cond_dim = condition.shape[1]
    target_dim = 1
    latent_dim = 16
    seq_len = 24
    d_model = 256
    nhead = 8
    num_layers = 3
    
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
    model.eval()
    
    # Initialize lists to store daily metrics
    daily_improvements = []
    daily_abs_diffs = []
    daily_rel_improvements = []
    daily_crps = []
    daily_pinball_losses = []
    all_samples_last_week = [] # Store samples for plotting if needed later
    all_targets_last_week = [] # Store targets for plotting

    # Determine indices for the last 7 days
    num_val_days = len(val_dataset)
    last_week_start_idx = max(0, num_val_days - 7)

    # Process each day in validation set
    for day_idx, (day_cond, day_target) in enumerate(val_loader):
        if (day_idx + 1) % 20 == 0:
            print(f"Processing validation day {day_idx+1}/{num_val_days}")

        day_cond = day_cond.to(device)
        day_target = day_target.to(device)

        # Generate samples for this day
        with torch.no_grad():
            samples = generate_samples(model, day_cond, num_samples, device) # Shape: [num_samples, batch=1, seq_len, 1]

        # Calculate metrics for this day
        improvement_metrics = evaluate_imbalance_improvement(day_target, samples)
        daily_improvements.append(improvement_metrics['improvement_percentage'])
        daily_abs_diffs.append(improvement_metrics['avg_absolute_diff'])
        daily_rel_improvements.append(improvement_metrics['relative_improvement'])

        crps = calculate_crps(day_target, samples)
        daily_crps.append(crps)

        pinball = calculate_pinball_losses(day_target, samples)
        daily_pinball_losses.append(pinball)

        # --- Output First Sample Prediction ---
        first_sample_pred = samples[0, :, :, :] # Select the first sample [1, seq_len, 1]
        first_sample_pred_np = first_sample_pred.squeeze().cpu().numpy() # [seq_len]
        target_np_for_print = day_target.squeeze().cpu().numpy() # [seq_len]
        print(f"--- Day {n_train + day_idx} ---")
        print(f" Target:   {np.round(target_np_for_print, 4)}")
        print(f" Sample 0: {np.round(first_sample_pred_np, 4)}")
        # ----------------------------------

        # --- Plotting logic for the last 7 days ---
        if day_idx >= last_week_start_idx:
            # Calculate 95th percentile of absolute predictions
            abs_samples = torch.abs(samples) # [num_samples, 1, seq_len, 1]
            conservative_pred = torch.quantile(abs_samples, 0.95, dim=0) # [1, seq_len, 1]

            # Prepare data for plotting
            target_np = day_target.squeeze().cpu().numpy() # [seq_len]
            pred_95_np = conservative_pred.squeeze().cpu().numpy() # [seq_len]
            hours = range(seq_len) # Typically 0-23

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(hours, target_np, label='Actual Imbalance', marker='o', linestyle='-')
            plt.plot(hours, pred_95_np, label='Predicted 95th Percentile (Absolute)', marker='x', linestyle='--')
            plt.xlabel('Hour of Day')
            plt.ylabel('Imbalance Value')
            # Use the absolute index in the original dataset for title
            original_day_index = n_train + day_idx
            plt.title(f'Validation Day {original_day_index}: Actual vs. Predicted 95th Percentile')
            plt.legend()
            plt.grid(True)
            plt.xticks(hours) # Ensure all hours are marked

            # Save plot
            plot_filename = os.path.join(output_dir, f'validation_day_{original_day_index}_plot.png')
            plt.savefig(plot_filename)
            plt.close() # Close the figure to free memory
        # -------------------------------------------

    # Calculate overall metrics for the entire validation set
    avg_improvement = np.mean(daily_improvements)
    avg_abs_diff = np.mean(daily_abs_diffs)
    avg_rel_improvement = np.mean(daily_rel_improvements)
    avg_crps = np.mean(daily_crps)
    
    # Calculate average pinball losses across all validation days
    avg_pinball = {}
    for key in daily_pinball_losses[0].keys():
        avg_pinball[key] = np.mean([day[key] for day in daily_pinball_losses])
    
    # Print results for the entire validation set
    print("\n--- Overall Validation Set Performance ---")
    print(f"Number of days evaluated: {len(val_dataset)}")
    print(f"Imbalance Reduction Metrics (95th percentile):")
    print(f"  Percentage of times model reduces imbalance: {avg_improvement:.2f}%")
    print(f"  Average absolute difference from actual: {avg_abs_diff:.4f}")
    print(f"  Average relative improvement: {avg_rel_improvement:.2f}%")
    print(f"Probabilistic Metrics:")
    print(f"  Average CRPS: {avg_crps:.6f}")
    for q, loss in sorted(avg_pinball.items()):
        print(f"  Average Pinball Loss ({q}): {loss:.6f}")
        
    # --- Calculate metrics for the last 7 days --- 
    if len(val_dataset) >= 7:
        last_week_improvements = np.mean(daily_improvements[-7:])
        last_week_abs_diffs = np.mean(daily_abs_diffs[-7:])
        last_week_rel_improvements = np.mean(daily_rel_improvements[-7:])
        last_week_crps = np.mean(daily_crps[-7:])
        
        last_week_pinball = {}
        for key in daily_pinball_losses[0].keys():
            last_week_pinball[key] = np.mean([day[key] for day in daily_pinball_losses[-7:]])
            
        # Print results for the last week
        print("\n--- Last 7 Days Validation Set Performance ---")
        print(f"Number of days evaluated: 7")
        print(f"Imbalance Reduction Metrics (95th percentile):")
        print(f"  Percentage of times model reduces imbalance: {last_week_improvements:.2f}%")
        print(f"  Average absolute difference from actual: {last_week_abs_diffs:.4f}")
        print(f"  Average relative improvement: {last_week_rel_improvements:.2f}%")
        print(f"Probabilistic Metrics:")
        print(f"  Average CRPS: {last_week_crps:.6f}")
        for q, loss in sorted(last_week_pinball.items()):
            print(f"  Average Pinball Loss ({q}): {loss:.6f}")
            
        last_week_metrics_dict = {
            "crps": last_week_crps,
            **last_week_pinball,
            "improvement_percentage": last_week_improvements,
            "average_absolute_difference": last_week_abs_diffs,
            "relative_improvement": last_week_rel_improvements
        }
    else:
        print("\nValidation set has less than 7 days, skipping last week analysis.")
        last_week_metrics_dict = None
    # ---------------------------------------------
    
    # Save detailed results
    results = {
        "daily_metrics": {
            "improvements": daily_improvements,
            "abs_diffs": daily_abs_diffs,
            "rel_improvements": daily_rel_improvements,
            "crps": daily_crps,
            "pinball_losses": daily_pinball_losses
        },
        "overall_validation_metrics": {
            "crps": avg_crps,
            **avg_pinball,
            "improvement_percentage": avg_improvement,
            "average_absolute_difference": avg_abs_diff,
            "relative_improvement": avg_rel_improvement
        },
        "last_week_validation_metrics": last_week_metrics_dict
    }
    
    np.save(os.path.join(output_dir, "validation_metrics.npy"), results)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")
    print(f"Plots for the last {min(7, num_val_days)} validation days saved in {output_dir}") # Added info about plots

if __name__ == "__main__":
    main() 