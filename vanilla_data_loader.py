import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class TrajectoryDataset(Dataset):
    """
    Dataloader for the Trajectory datasets
    """
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, plot_trajectories=False):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
          <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while sampling sequences.
        - plot_trajectories: If True, will display a plot of all raw trajectories.
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        # Get all file paths in the data directory
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        # This will hold all the processed trajectory data
        all_ped_trajectories = []
        
        print(f"Processing files in: {self.data_dir}")
        for path in all_files:
            print(f"  - Loading {os.path.basename(path)}")
            # Load data from the text file
            # The format is: frame_id, ped_id, x, y
            data = np.loadtxt(path, delimiter='\t')
            
            # Group trajectories by pedestrian ID
            peds = np.unique(data[:, 1])
            for ped_id in peds:
                # Get all rows for the current pedestrian
                ped_data = data[data[:, 1] == ped_id, :]
                # Store the trajectory [x, y]
                all_ped_trajectories.append(ped_data[:, 2:])
        
        # Plot all trajectories if requested
        if plot_trajectories:
            self._plot_all_trajectories(all_ped_trajectories)

        # Now, create observation/prediction sequences
        self.obs_traj = []
        self.pred_traj = []
        
        for traj in all_ped_trajectories:
            # Check if the trajectory is long enough to create a sequence
            if len(traj) < self.seq_len:
                continue
            
            # Create sliding window sequences
            num_sequences = (len(traj) - self.seq_len) // self.skip + 1
            for i in range(0, num_sequences * self.skip, self.skip):
                # Observation sequence
                obs = traj[i : i + self.obs_len, :]
                # Prediction sequence
                pred = traj[i + self.obs_len : i + self.seq_len, :]
                
                self.obs_traj.append(obs)
                self.pred_traj.append(pred)

        # Convert to PyTorch Tensors
        self.obs_traj = torch.tensor(np.array(self.obs_traj), dtype=torch.float32)
        self.pred_traj = torch.tensor(np.array(self.pred_traj), dtype=torch.float32)
        print(f"Total sequences processed: {len(self.obs_traj)}")

    def _plot_all_trajectories(self, all_ped_trajectories):
        """
        Helper function to plot all raw trajectories.
        """
        print("Displaying plot of all raw trajectories...")
        plt.figure(figsize=(10, 8))
        for traj in all_ped_trajectories:
            # Plot each trajectory with a unique color automatically
            plt.plot(traj[:, 0], traj[:, 1])
        
        plt.title('All Raw Pedestrian Trajectories (Zara01 Dataset)')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is equal
        plt.show()

    def __len__(self):
        return len(self.obs_traj)

    def __getitem__(self, index):
        return self.obs_traj[index], self.pred_traj[index]


# --- Example Usage ---
if __name__ == '__main__':
    train_data_dir = './datasets/zara1/train'
    
    # Check if the directory and at least one .txt file exist
    if not os.path.exists(train_data_dir) or not any(f.endswith('.txt') for f in os.listdir(train_data_dir)):
        print(f"Error: Make sure the directory '{train_data_dir}' exists and contains your dataset .txt file (e.g., crowds_zara01.txt).")
        print("Please download the Zara01 dataset and place it in the correct directory.")
    else:
        print("Found dataset file. Initializing DataLoader...")
        # Define parameters
        obs_len = 8
        pred_len = 12
        batch_size = 64

        # Create dataset object and generate the plot
        train_dataset = TrajectoryDataset(
            data_dir=train_data_dir,
            obs_len=obs_len,
            pred_len=pred_len,
            plot_trajectories=True  # This flag will trigger the plot
        )

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Fetch one batch to verify
        try:
            obs_batch, pred_batch = next(iter(train_loader))
            print("\nSuccessfully loaded one batch of data!")
            print(f"Observation batch shape: {obs_batch.shape}")
            print(f"Prediction batch shape: {pred_batch.shape}")
            # Expected shapes:
            # obs_batch: (batch_size, obs_len, 2)
            # pred_batch: (batch_size, pred_len, 2)
        except StopIteration:
            print("\nCould not load a batch. The dataset might be too small or empty.")

