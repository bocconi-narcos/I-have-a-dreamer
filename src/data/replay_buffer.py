import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ReplayBufferDataset(Dataset):
    """
    Unified dataset for sequential decision-making training. Expects a buffer (list of dicts) with the following keys:
        - state: (H, W) ndarray or tensor
        - target_state: (H, W)
        - color_in_state: int
        - action: dict with keys 'colour', 'selection', 'transform'
        - colour: int (the true color after selection)
        - selection_mask: (H, W)
        - reward: float
        - next_state: (H, W)
        - done: bool
        - info: dict (optional)
    If buffer_path does not exist, generates dummy data with the correct structure.
    
    Args:
        buffer_path (str): Path to the replay buffer file
        num_color_selection_fns (int): Number of color selection functions
        num_selection_fns (int): Number of selection functions  
        num_transform_actions (int): Number of transform actions
        num_arc_colors (int): Number of ARC colors
        state_shape (tuple): Shape of the state (C, H, W) or (H, W)
        num_samples (int): Number of samples to generate if buffer doesn't exist
        mode (str): Training mode - 'color_only', 'selection_color', or 'end_to_end'
    """
    def __init__(self, buffer_path, num_color_selection_fns, num_selection_fns, num_transform_actions, 
                 num_arc_colors, state_shape, num_samples=1000, mode='end_to_end'):
        self.num_color_selection_fns = num_color_selection_fns
        self.num_selection_fns = num_selection_fns
        self.num_transform_actions = num_transform_actions
        self.num_arc_colors = num_arc_colors
        self.state_shape = state_shape
        self.mode = mode
        
        if os.path.exists(buffer_path):
            # TODO: Implement real buffer loading
            # Example: with open(buffer_path, 'rb') as f: self.buffer = pickle.load(f)
            raise NotImplementedError("Buffer loading not implemented yet.")
        else:
            # Generate dummy data with the correct structure
            self.buffer = []
            for _ in range(num_samples):
                state = np.random.randint(0, num_arc_colors, size=state_shape).astype(np.int64)
                target_state = np.random.randint(0, num_arc_colors, size=state_shape).astype(np.int64)
                color_in_state = np.random.randint(1, num_arc_colors+1)
                action = {
                    'colour': np.random.randint(0, num_color_selection_fns),
                    'selection': np.random.randint(0, num_selection_fns),
                    'transform': np.random.randint(0, num_transform_actions)
                }
                colour = np.random.randint(0, num_arc_colors)
                selection_mask = np.random.randint(0, 2, size=state_shape).astype(np.int64)
                reward = np.random.uniform(-1, 1)
                next_state = np.random.randint(0, num_arc_colors, size=state_shape).astype(np.int64)
                done = np.random.choice([True, False])
                info = {}
                self.buffer.append({
                    'state': state,
                    'target_state': target_state,
                    'color_in_state': color_in_state,
                    'action': action,
                    'colour': colour,
                    'selection_mask': selection_mask,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'info': info
                })
            self.num_samples = num_samples

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        sample = self.buffer[idx]
        
        # Convert to torch tensors
        state = torch.tensor(sample['state'], dtype=torch.long)
        action_colour = torch.tensor(sample['action']['colour'], dtype=torch.long)
        action_selection = torch.tensor(sample['action']['selection'], dtype=torch.long)
        action_transform = torch.tensor(sample['action']['transform'], dtype=torch.long)
        colour = torch.tensor(sample['colour'], dtype=torch.long)
        selection_mask = torch.tensor(sample['selection_mask'], dtype=torch.float)
        
        # Base return dict with common fields
        result = {
            'state': state,  # (H, W) or (C, H, W)
            'action_colour': action_colour,  # int
            'action_selection': action_selection,  # int
            'action_transform': action_transform,  # int
            'colour': colour,  # int
            'selection_mask': selection_mask  # (H, W) or (C, H, W)
        }
        
        # Add mode-specific fields
        if self.mode in ['selection_color', 'end_to_end']:
            # Add next_state for selection_color and end_to_end modes
            next_state = torch.tensor(sample['next_state'], dtype=torch.long)
            result['next_state'] = next_state
        
        return result 