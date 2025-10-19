import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import TrajectoryDataset  # Import the dataset class
from vanilla_lstm_model import VanillaLSTM # Import the model class
import matplotlib.pyplot as plt
import os

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
    num_layers = 20

    # Training parameters
    learning_rate = 0.0005
    num_epochs = 50 
    
    # Directory to save plots
    plot_dir = './plots_{embedding_dim}_{hidden_dim}_{num_layers}/train_{learning_rate}_{num_epochs}'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- 2. Data Loading ---
    print("Loading data...")
    train_data_dir = './datasets/zara1/train'
    dataset = TrajectoryDataset(data_dir=train_data_dir, obs_len=obs_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Model, Loss, and Optimizer Initialization ---
    model = VanillaLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    loss_fn = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 4. Training Loop ---
    print("Starting training...")
    loss_history = [] # To store loss per epoch for plotting
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        
        for batch_idx, (obs_traj, pred_traj_true) in enumerate(loader):
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            optimizer.zero_grad()
            pred_traj_fake = model(obs_traj, pred_len=pred_len)
            loss = loss_fn(pred_traj_fake, pred_traj_true)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # --- 5. Evaluation ---
    print("Evaluating model...")
    model.eval() 
    total_ade = 0
    total_fde = 0
    
    with torch.no_grad():
        for obs_traj, pred_traj_true in loader:
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            pred_traj_fake = model(obs_traj, pred_len=pred_len)
            total_ade += ade(pred_traj_fake, pred_traj_true).item()
            total_fde += fde(pred_traj_fake, pred_traj_true).item()
            
    avg_ade = total_ade / len(loader)
    avg_fde = total_fde / len(loader)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f}")
    print(f"Final Displacement Error (FDE): {avg_fde:.4f}")
    
    # --- 6. Plotting Training Loss ---
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'training_loss.png'))
    print(f"\nTraining loss plot saved to {os.path.join(plot_dir, 'training_loss.png')}")
    # plt.show() # Uncomment to display plot interactively

    # --- 7. Visualizing a Batch of Predictions ---
    print("Generating evaluation plot...")
    with torch.no_grad():
        # Get one batch from the loader
        obs_traj, pred_traj_true = next(iter(loader))
        obs_traj = obs_traj.to(device)
        pred_traj_fake = model(obs_traj, pred_len=pred_len).cpu()

        # Detach tensors and move to cpu for plotting
        obs_traj = obs_traj.cpu()
        pred_traj_true = pred_traj_true.cpu()
        
        plt.figure(figsize=(12, 10))
        # Plot 5 examples from the batch
        for i in range(5):
            plt.subplot(3, 2, i + 1)
            
            # Plot observed trajectory (past)
            plt.plot(obs_traj[i, :, 0], obs_traj[i, :, 1], 'b-o', label='Observed Path')
            
            # Plot true future trajectory
            plt.plot(pred_traj_true[i, :, 0], pred_traj_true[i, :, 1], 'g-s', label='True Future')

            # Plot predicted future trajectory
            plt.plot(pred_traj_fake[i, :, 0], pred_traj_fake[i, :, 1], 'r-x', label='Predicted Future')

            plt.title(f'Example {i+1}')
            plt.axis('equal')
            plt.grid(True)
        
        plt.suptitle('Sample Evaluation Trajectories')
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.savefig(os.path.join(plot_dir, 'evaluation_examples.png'))
        print(f"Evaluation examples plot saved to {os.path.join(plot_dir, 'evaluation_examples.png')}")

