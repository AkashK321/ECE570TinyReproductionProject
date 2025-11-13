import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def detect_groups_at_frame(frame_coords, distance_threshold=1.0, min_group_size=2):
    """Runs DBSCAN and returns group labels and group centers."""
    if len(frame_coords) == 0:
        return np.array([]), {}
        
    db = DBSCAN(eps=distance_threshold, min_samples=min_group_size).fit(frame_coords)
    labels = db.labels_
    
    group_centers = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label != -1: # -1 is for lone individuals
            # Get coords for all members of this group
            group_coords = frame_coords[labels == label]
            # Calculate the center (mean) of the group
            group_centers[label] = np.mean(group_coords, axis=0)
            
    return labels, group_centers

class TrajectoryDataset(Dataset):
    """
    Dataloader for the Trajectory datasets
    UPDATED to pre-process and include group information.
    """
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, plot_trajectories=False):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        all_ped_trajectories = [] # Stores individual trajectories
        ped_id_to_traj_idx = {}   # Maps a pedestrian ID to their index in all_ped_trajectories
        frame_to_peds = {}        # Maps a frame ID to list of [ped_id, x, y]
        
        print(f"Processing files in: {self.data_dir}")
        current_traj_idx = 0
        all_raw_data = []
        for path in all_files:
            print(f"  - Loading {os.path.basename(path)}")
            data = np.loadtxt(path, delimiter='\t')
            all_raw_data.append(data)
            
            ped_ids = np.unique(data[:, 1])

            # 1. Process individual trajectories and frame-to-ped mapping
            for ped_id in ped_ids:
                ped_data = data[data[:, 1] == ped_id, :]
                all_ped_trajectories.append(ped_data) # Store full data [frame, id, x, y]
                ped_id_to_traj_idx[ped_id] = current_traj_idx
                current_traj_idx += 1

        # Combine all raw data to process frames
        if not all_raw_data:
             print("No data loaded. Exiting.")
             return
             
        full_raw_data = np.concatenate(all_raw_data, axis=0)
        frame_ids = np.unique(full_raw_data[:, 0])

        for frame_id in frame_ids:
            frame_data = full_raw_data[full_raw_data[:, 0] == frame_id, :]
            # Store [ped_id, x, y]
            frame_to_peds[frame_id] = frame_data[:, [1, 2, 3]]
                
        # 2. Run group detection on every frame
        print("Detecting groups in all frames...")
        # Stores group center for each ped at each frame
        # We'll build a map: (ped_id, frame_id) -> (group_center_x, group_center_y)
        self.group_center_map = {}
        
        for frame_id, peds_in_frame in frame_to_peds.items():
            if len(peds_in_frame) < 2:
                for ped_id, x, y in peds_in_frame:
                    self.group_center_map[(ped_id, frame_id)] = (x, y) # Lone person's group center is themself
                continue
                
            ped_ids = peds_in_frame[:, 0]
            coords = peds_in_frame[:, 1:]
            
            labels, group_centers = detect_groups_at_frame(coords, distance_threshold=1.0, min_group_size=2)
            
            for i, ped_id in enumerate(ped_ids):
                label = labels[i]
                if label == -1: # Lone individual
                    self.group_center_map[(ped_id, frame_id)] = coords[i] # Group center is themself
                else: # Part of a group
                    self.group_center_map[(ped_id, frame_id)] = group_centers[label]
        
        print("Group detection complete.")
        
        # 3. Create observation/prediction sequences
        self.obs_traj = []
        self.pred_traj = []
        self.obs_group_traj = [] # NEW: Store group trajectories
        
        for ped_full_data in all_ped_trajectories:
            if len(ped_full_data) < self.seq_len:
                continue
            
            num_sequences = (len(ped_full_data) - self.seq_len) // self.skip + 1
            for i in range(0, num_sequences * self.skip, self.skip):
                # --- Individual Trajectory ---
                obs_data = ped_full_data[i : i + self.obs_len]
                pred_data = ped_full_data[i + self.obs_len : i + self.seq_len]
                
                self.obs_traj.append(obs_data[:, 2:]) # (x, y)
                self.pred_traj.append(pred_data[:, 2:]) # (x, y)
                
                # --- Group Trajectory (NEW) ---
                current_obs_group_traj = []
                valid_sequence = True
                for frame_id, ped_id, _, _ in obs_data:
                    # Look up the pre-computed group center for this ped at this frame
                    group_center = self.group_center_map.get((ped_id, frame_id))
                    if group_center is None:
                        # This should not happen if all frames are processed, but as a fallback:
                        valid_sequence = False
                        break
                    current_obs_group_traj.append(group_center)
                
                if valid_sequence:
                    self.obs_group_traj.append(np.array(current_obs_group_traj))
                else:
                    # If sequence was invalid, remove the corresponding obs/pred
                    self.obs_traj.pop()
                    self.pred_traj.pop()

        # Convert to PyTorch Tensors
        self.obs_traj = torch.tensor(np.array(self.obs_traj), dtype=torch.float32)
        self.pred_traj = torch.tensor(np.array(self.pred_traj), dtype=torch.float32)
        self.obs_group_traj = torch.tensor(np.array(self.obs_group_traj), dtype=torch.float32)
        
        print(f"Total sequences processed: {len(self.obs_traj)}")
        
        # Plot all raw trajectories (from ped_full_data) if requested
        # Note: This is less efficient as it plots after processing, but fine for debug
        if plot_trajectories:
             # This plots individual trajectories, not groups
             self._plot_all_trajectories([traj[:, 2:] for traj in all_ped_trajectories])


    def _plot_all_trajectories(self, all_ped_trajectories):
        """
        Helper function to plot all raw trajectories.
        """
        print("Displaying plot of all raw trajectories...")
        plt.figure(figsize=(10, 8))
        for traj in all_ped_trajectories:
            # Plot each trajectory with a unique color automatically
            plt.plot(traj[:, 0], traj[:, 1])
        
        plt.title('All Raw Pedestrian Trajectories')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is equal
        plt.show()

    def __len__(self):
        # Ensure all lists are the same length
        return len(self.obs_traj)

    def __getitem__(self, index):
        # Return all three tensors
        return self.obs_traj[index], self.pred_traj[index], self.obs_group_traj[index]


# --- Example Usage (to test the loader) ---
if __name__ == '__main__':
    train_data_dir = './datasets/zara1/train'
    
    if not os.path.exists(train_data_dir) or not any(f.endswith('.txt') for f in os.listdir(train_data_dir)):
        print(f"Error: Make sure the directory '{train_data_dir}' exists and contains your dataset .txt file.")
    else:
        print("Found dataset file. Initializing DataLoader (with group processing)...")
        dataset = TrajectoryDataset(
            data_dir=train_data_dir,
            obs_len=8,
            pred_len=12
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        try:
            obs_batch, pred_batch, obs_group_batch = next(iter(loader))
            print("\nSuccessfully loaded one batch of data!")
            print(f"Observation batch shape: {obs_batch.shape}")
            print(f"Prediction batch shape: {pred_batch.shape}")
            print(f"Group Observation batch shape: {obs_group_batch.shape}")
        except StopIteration:
            print("\nCould not load a batch. The dataset might be too small or empty.")
        except ValueError:
            print("\nError loading batch. Is __getitem__ returning 3 items?")
            # This is a good place to double-check the loader's state
            print(f"Loaded obs_traj: {len(dataset.obs_traj)}")
            print(f"Loaded pred_traj: {len(dataset.pred_traj)}")
            print(f"Loaded obs_group_traj: {len(dataset.obs_group_traj)}")