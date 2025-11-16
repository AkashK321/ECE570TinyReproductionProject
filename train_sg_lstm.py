import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sg_data_loader import TrajectoryDataset  # Import the UPDATED dataset class
from sg_lstm_model import SGLSTM           # Import the NEW model class
import matplotlib.pyplot as plt
import os
import time

# --- Helper Functions for Evaluation (ADE and FDE) ---
# (These are identical to the ones in your previous train.py)
def ade(pred, true):
    diff = pred - true
    dist = torch.sqrt(torch.sum(diff**2, dim=-1))
    return torch.mean(dist)

def fde(pred, true):
    final_pred = pred[:, -1, :]
    final_true = true[:, -1, :]
    diff = final_pred - final_true
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
    embedding_dim = 64 # Embedding size for BOTH individual and group
    hidden_dim = 128   # Hidden dim (might need to be larger to handle combined info)
    num_layers = 10

    # Training parameters
    learning_rate = 0.0005
    num_epochs = 50 
    
    plot_dir = './plots_sg_lstm'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- 2. Data Loading (using the updated loader) ---
    print("Loading data (with group processing)...")
    train_data_dir = './datasets/zara1/train'
    dataset = TrajectoryDataset(data_dir=train_data_dir, obs_len=obs_len, pred_len=pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 3. Model, Loss, and Optimizer Initialization ---
    model = SGLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    loss_fn = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 4. Training Loop ---
    print("Starting SG-LSTM training...")
    loss_history = []
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        
        # The loader now yields three items
        for batch_idx, (obs_traj, pred_traj_true, obs_group_traj) in enumerate(loader):
            # Move all data to the device
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            obs_group_traj = obs_group_traj.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with both individual and group trajectories
            pred_traj_fake = model(obs_traj, obs_group_traj, pred_len=pred_len)
            
            # Loss is still computed against the individual's true path
            loss = loss_fn(pred_traj_fake, pred_traj_true)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(loader)
        loss_history.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # --- 5. Evaluation ---
    print("Evaluating SG-LSTM model...")
    eval_start_time = time.time()
    model.eval() 
    total_ade = 0
    total_fde = 0
    
    with torch.no_grad():
        for obs_traj, pred_traj_true, obs_group_traj in loader:
            obs_traj = obs_traj.to(device)
            pred_traj_true = pred_traj_true.to(device)
            obs_group_traj = obs_group_traj.to(device)
            
            pred_traj_fake = model(obs_traj, obs_group_traj, pred_len=pred_len)
            
            total_ade += ade(pred_traj_fake, pred_traj_true).item()
            total_fde += fde(pred_traj_fake, pred_traj_true).item()
            
    avg_ade = total_ade / len(loader)
    avg_fde = total_fde / len(loader)

    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time
    
    print(f"\n--- SG-LSTM Evaluation Results ---")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f}")
    print(f"Final Displacement Error (FDE): {avg_fde:.4f}")
    print(f"Evaluation took: {eval_duration:.2f} seconds")
    
    # --- 6. Plotting Training Loss ---
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.title('SG-LSTM Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'sg_lstm_training_loss.png'))
    print(f"\nTraining loss plot saved to {os.path.join(plot_dir, 'sg_lstm_training_loss.png')}")
    # plt.show() 

    # --- 7. Visualizing a Batch of Predictions ---
    print("Generating SG-LSTM evaluation plot...")
    with torch.no_grad():
        obs_traj, pred_traj_true, obs_group_traj = next(iter(loader))
        obs_traj = obs_traj.to(device)
        obs_group_traj = obs_group_traj.to(device)
        pred_traj_fake = model(obs_traj, obs_group_traj, pred_len=pred_len).cpu()

        obs_traj = obs_traj.cpu()
        pred_traj_true = pred_traj_true.cpu()
        obs_group_traj = obs_group_traj.cpu() # Get group data for plotting
        
        plt.figure(figsize=(12, 10))
        for i in range(5):
            plt.subplot(3, 2, i + 1)
            
            # Plot observed trajectory (past)
            plt.plot(obs_traj[i, :, 0], obs_traj[i, :, 1], 'b-o', label='Observed (Indiv)')
            # Plot observed group center (past)
            plt.plot(obs_group_traj[i, :, 0], obs_group_traj[i, :, 1], 'c--o', label='Observed (Group)')
            
            # Plot true future trajectory
            plt.plot(pred_traj_true[i, :, 0], pred_traj_true[i, :, 1], 'g-s', label='True Future')

            # Plot predicted future trajectory
            plt.plot(pred_traj_fake[i, :, 0], pred_traj_fake[i, :, 1], 'r-x', label='Predicted Future')

            plt.title(f'Example {i+1}')
            plt.axis('equal')
            plt.grid(True)
        
        plt.suptitle('Sample SG-LSTM Evaluation Trajectories')
        handles, labels_list = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_list, handles)) # remove duplicate labels
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.savefig(os.path.join(plot_dir, 'sg_lstm_evaluation_examples.png'))
        print(f"Evaluation examples plot saved to {os.path.join(plot_dir, 'sg_lstm_evaluation_examples.png')}")
        plt.show()