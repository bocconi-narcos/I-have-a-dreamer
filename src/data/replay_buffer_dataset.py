import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import os


class ReplayBufferDataset(Dataset):
    """
    Dataset for loading replay buffer data for training sequential decision-making modules.
    
    Args:
        buffer_path (str): Path to the replay buffer pickle file
        num_color_selection_fns (int): Number of color selection functions
        num_selection_fns (int): Number of selection functions  
        num_transform_actions (int): Number of transform actions
        num_arc_colors (int): Number of ARC colors
        state_shape (tuple): Expected shape of state tensors (C, H, W)
        mode (str): Training mode - 'color_only', 'selection_color', or 'end_to_end'
        num_samples (int, optional): Number of samples to use (for testing). If None, uses all data.
    """
    
    def __init__(self, buffer_path, num_color_selection_fns, num_selection_fns, 
                 num_transform_actions, num_arc_colors, state_shape, mode='color_only', num_samples=None):
        self.buffer_path = buffer_path
        self.num_color_selection_fns = num_color_selection_fns
        self.num_selection_fns = num_selection_fns
        self.num_transform_actions = num_transform_actions
        self.num_arc_colors = num_arc_colors
        self.state_shape = state_shape
        self.mode = mode
        
        # Load or generate buffer data
        if os.path.exists(buffer_path):
            print(f"Loading replay buffer from {buffer_path}")
            with open(buffer_path, 'rb') as f:
                self.buffer = pickle.load(f)
        else:
            print(f"ERROR: Buffer file {buffer_path} not found. Please provide a valid replay buffer file.")
            raise FileNotFoundError(f"Buffer file {buffer_path} not found.")
        
        # Limit samples if specified (for testing)
        if num_samples is not None:
            self.buffer = self.buffer[:num_samples]
        
        print(f"Dataset initialized with {len(self.buffer)} samples in {mode} mode")
    
    def _generate_dummy_data(self):
        """Generate dummy replay buffer data for testing."""
        num_samples = 1000
        buffer = []
        
        for i in range(num_samples):
            # Generate random state
            if len(self.state_shape) == 3:
                C, H, W = self.state_shape
                state = np.random.randint(0, self.num_arc_colors, (H, W))
            else:
                H, W = self.state_shape
                state = np.random.randint(0, self.num_arc_colors, (H, W))
            
            # Generate random actions
            action = {
                'colour': np.random.randint(0, self.num_color_selection_fns),
                'selection': np.random.randint(0, self.num_selection_fns),
                'transform': np.random.randint(0, self.num_transform_actions)
            }
            
            # Generate random selection mask
            selection_mask = np.random.randint(0, 2, state.shape)
            
            # Generate random next state
            next_state = np.random.randint(0, self.num_arc_colors, state.shape)
            
            # Generate random color result
            colour = np.random.randint(0, self.num_arc_colors)
            
            # Create transition
            transition = {
                'state': state,
                'action': action,
                'selection_mask': selection_mask,
                'next_state': next_state,
                'colour': colour,
                'reward': np.random.random(),
                'done': np.random.choice([True, False]),
                'info': {}
            }
            
            buffer.append(transition)
        
        return buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        transition = self.buffer[idx]
        
        # Extract state and convert to tensor
        state = torch.tensor(transition['state'], dtype=torch.float32)
        
        # Extract actions
        action_colour = torch.tensor(transition['action']['colour'], dtype=torch.long)
        action_selection = torch.tensor(transition['action']['selection'], dtype=torch.long)
        action_transform = torch.tensor(transition['action']['transform'], dtype=torch.long)
        
        # Extract targets
        colour = torch.tensor(transition['colour'], dtype=torch.long)
        selection_mask = torch.tensor(transition['selection_mask'], dtype=torch.float32)
        
        # Prepare sample based on mode
        sample = {
            'state': state,
            'action_colour': action_colour,
            'action_selection': action_selection,
            'action_transform': action_transform,
            'colour': colour,
            'selection_mask': selection_mask,
        }
        
        # Add next_state for modes that need it
        if self.mode in ['selection_color', 'end_to_end']:
            next_state = torch.tensor(transition['next_state'], dtype=torch.float32)
            sample['next_state'] = next_state
        
        # Add transition_type if present
        transition_type = transition.get('info', {}).get('transition_type', None)
        sample['transition_type'] = transition_type
        
        return sample 