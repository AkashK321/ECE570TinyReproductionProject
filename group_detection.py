import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

def get_frame_data(data, frame_id):
    """Extracts all (x, y) coordinates for a specific frame."""
    # Get all rows where the frame_id matches
    frame_data = data[data[:, 0] == frame_id]
    # Return just the x, y coordinates
    return frame_data[:, 2:]

def detect_groups(frame_coords, distance_threshold=1.0, min_group_size=2):
    """
    Runs DBSCAN to cluster pedestrians into groups.
    
    Args:
    - frame_coords: (N, 2) array of (x, y) coordinates for N pedestrians
    - distance_threshold: (eps) Max distance to be considered a neighbor.
    - min_group_size: Min pedestrians to form a group.
    
    Returns:
    - labels: (N,) array of group labels. -1 indicates a lone individual (noise).
    """
    # Create and run DBSCAN
    # eps = max distance between two samples for one to be considered
    # as in the neighborhood of the other. This is our social distance.
    # min_samples = The number of samples in a neighborhood for a point
    # to be considered as a core point. This is our min group size.
    db = DBSCAN(eps=distance_threshold, min_samples=min_group_size).fit(frame_coords)
    return db.labels_

def plot_groups_at_frame(frame_coords, labels, frame_id, plot_dir='./plots'):
    """
    Generates and saves a plot of the detected groups for a specific frame.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.figure(figsize=(12, 10))
    
    # Get the unique group labels. -1 is for noise (lone individuals).
    unique_labels = set(labels)
    
    # Create a color map
    # Fix for deprecation warning and TypeError:
    # 1. Use colormaps.get_cmap() instead of cm.get_cmap()
    # 2. Call the colormap with np.linspace to get an iterable list of colors
    cmap = plt.colormaps.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for lone individuals
            col = [0, 0, 0, 1]
            label = 'Lone Individual'
        else:
            label = f'Group {k+1}'

        # Get all points belonging to this cluster
        class_member_mask = (labels == k)
        xy = frame_coords[class_member_mask]
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10, label=label)

    plt.title(f'Heuristic Group Detection (DBSCAN) on Frame {frame_id}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Only show legend once if there are many pedestrians
    handles, labels_list = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles)) # remove duplicate labels
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True)
    plt.axis('equal')
    
    save_path = os.path.join(plot_dir, f'group_detection_frame_{int(frame_id)}.png')
    plt.savefig(save_path)
    print(f"Group detection plot saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # This script demonstrates the group detection heuristic
    
    # --- 1. Load Raw Data ---
    # We load the raw data, not the processed data, as we need per-frame info
    data_file_path = './datasets/zara1/train/students001_train.txt' # Adjust to your file
    if not os.path.exists(data_file_path):
        print(f"Error: Could not find raw data file at {data_file_path}")
    else:
        print("Loading raw data...")
        raw_data = np.loadtxt(data_file_path, delimiter='\t')
        
        # --- 2. Pick a Frame and Get Data ---
        # Let's pick a frame we know is busy from the previous plot
        target_frame_id = 70.0 
        ped_coords = get_frame_data(raw_data, target_frame_id)
        
        if len(ped_coords) == 0:
            print(f"No data found for frame {target_frame_id}. Try a different frame.")
        else:
            print(f"Found {len(ped_coords)} pedestrians in frame {target_frame_id}.")
            
            # --- 3. Run Group Detection ---
            # We'll assume a "group" means people are within 1.0 meters of each other
            group_labels = detect_groups(ped_coords, distance_threshold=1.0, min_group_size=2)
            
            num_groups = len(set(group_labels)) - (1 if -1 in group_labels else 0)
            num_lone = np.sum(group_labels == -1)
            print(f"Detected {num_groups} groups and {num_lone} lone individuals.")
            
            # --- 4. Plot Results ---
            plot_groups_at_frame(ped_coords, group_labels, target_frame_id)
