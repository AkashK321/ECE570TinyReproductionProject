import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import TrajectoryDataset  # Import the dataset class
from vanilla_lstm_model import VanillaLSTM # Import the model class

# --- Helper Functions for Evaluation ---
def ade(pred, true):
    """
    Calculates Average Displacement Error (L2 distance).
    """
    # Shape of pred/true: (batch_size, pred_len, 2)
    # Calculate difference
    diff = pred - true
    # Calculate L2 norm (Euclidean distance) along the coordinate dimension
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    # Average over the prediction length and then the batch
    return torch.mean(dist)

def fde(pred, true):
    """
    Calculates Final Displacement Error (L2 distance at the last step).
    """
    # Shape of pred/true: (batch_size, pred_len, 2)
    # Get the final positions
    final_pred = pred[:, -1, :]
    final_true = true[:, -1, :]
    # Calculate difference
    diff = final_pred - final_true
    # Calculate L2 norm and average over the batch
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    return torch.mean(dist)


# --- Main Training Block ---
if __name__ == '__main__':
    # --- 1. Hyperparameters and Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data parameters
    obs_len = 8
    pred_len = 12
    batch_size = 64
    
    # Model parameters
    embedding_dim = 64
    hidden_dim = 64
    num_layers = 1

    # Training parameters
    learning_rate = 0.001
    num_epochs = 50 # Start with a reasonable number of epochs

    # --- 2. Data Loading ---
    print("Loading data...")
    train_data_dir = './datasets/zara1/train'
    dataset = TrajectoryDataset(data_dir=train_data_dir, obs_len=obs_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Model, Loss, and Optimizer Initialization ---
    model = VanillaLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    loss_fn = nn.MSELoss() # Mean Squared Error is a good choice for coordinate regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        
        for batch_idx, (obs_traj, pred_traj_true) in enumerate(loader):
            # Move data to the device
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_traj_fake = model(obs_traj, pred_len=pred_len)
            
            # Calculate the loss
            loss = loss_fn(pred_traj_fake, pred_traj_true)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # --- 5. Evaluation ---
    print("Evaluating model...")
    model.eval() # Set model to evaluation mode
    total_ade = 0
    total_fde = 0
    
    with torch.no_grad():
        for obs_traj, pred_traj_true in loader:
            # Move data to the device
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            
            # Forward pass
            pred_traj_fake = model(obs_traj, pred_len=pred_len)
            
            # Calculate ADE and FDE for the batch
            total_ade += ade(pred_traj_fake, pred_traj_true).item()
            total_fde += fde(pred_traj_fake, pred_traj_true).item()
            
    # Calculate average metrics
    avg_ade = total_ade / len(loader)
    avg_fde = total_fde / len(loader)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f}")
    print(f"Final Displacement Error (FDE): {avg_fde:.4f}")