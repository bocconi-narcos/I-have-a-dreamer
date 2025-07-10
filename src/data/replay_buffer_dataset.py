import torch
import pickle
from torch.utils.data import Dataset
import os
import h5py


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
            if buffer_path.endswith('.hdf5'):
                self.buffer = self._load_hdf5_buffer(buffer_path)
            elif buffer_path.endswith('.pkl'):
                with open(buffer_path, 'rb') as f:
                    self.buffer = pickle.load(f)
            elif buffer_path.endswith('.pt'):
                import time
                start_time = time.time()
                buffer_dict = torch.load(buffer_path, map_location='cpu')
                
                # Convert dictionary format to list of transitions
                if isinstance(buffer_dict, dict) and 'state' in buffer_dict:
                    # This is a dictionary format with arrays for each field
                    num_transitions = len(buffer_dict['state'])
                    self.buffer = []
                    
                    for i in range(num_transitions):
                        transition = {
                            'state': buffer_dict['state'][i],
                            'action': {
                                'colour': buffer_dict['action_colour'][i],
                                'selection': buffer_dict['action_selection'][i],
                                'transform': buffer_dict['action_transform'][i]
                            },
                            'selection_mask': buffer_dict['selection_mask'][i],
                            'next_state': buffer_dict['next_state'][i],
                            'target_state': buffer_dict['target_state'][i],
                            'colour': buffer_dict['colour'][i],
                            'reward': buffer_dict['reward'][i],
                            'done': buffer_dict['done'][i],
                            'transition_type': buffer_dict['transition_type'][i],
                            'shape_h': buffer_dict['shape_h'][i],
                            'shape_w': buffer_dict['shape_w'][i],
                            'num_colors_grid': buffer_dict['num_colors_grid'][i],
                            'most_present_color': buffer_dict['most_present_color'][i],
                            'least_present_color': buffer_dict['least_present_color'][i],
                            'num_colors_grid_target': buffer_dict['num_colors_grid_target'][i],
                            'most_present_color_target': buffer_dict['most_present_color_target'][i],
                            'least_present_color_target': buffer_dict['least_present_color_target'][i],
                            'shape_h_target': buffer_dict['shape_h_target'][i],
                            'shape_w_target': buffer_dict['shape_w_target'][i],
                            'shape_h_next': buffer_dict['shape_h_next'][i],
                            'shape_w_next': buffer_dict['shape_w_next'][i],
                            'num_colors_grid_next': buffer_dict['num_colors_grid_next'][i],
                            'most_present_color_next': buffer_dict['most_present_color_next'][i],
                            'least_present_color_next': buffer_dict['least_present_color_next'][i]
                        }
                        self.buffer.append(transition)
                else:
                    # This is already in list format
                    self.buffer = buffer_dict
                
                end_time = time.time()
                print(f"Loaded {len(self.buffer)} transitions in {end_time - start_time:.2f} seconds")
            elif buffer_path.endswith('h5'):
                import time
                start_time = time.time()
                self.buffer = self._load_hdf5_buffer(buffer_path)
                end_time = time.time()
                print(f"Loaded {len(self.buffer)} transitions in {end_time - start_time:.2f} seconds")
            else:
                raise ValueError(f"Unsupported buffer file format: {buffer_path}. Please use .hdf5, .pkl, or .pt")
        else:
            print(f"ERROR: Buffer file {buffer_path} not found. Please provide a valid replay buffer file.")
            raise FileNotFoundError(f"Buffer file {buffer_path} not found.")
        
        # Limit samples if specified (for testing)
        if num_samples is not None:
            self.buffer = self.buffer[:num_samples]
        
        print(f"Dataset initialized with {len(self.buffer)} samples in {mode} mode")
    

    
    def _load_hdf5_buffer(self, buffer_path):
        """Load replay buffer data from an HDF5 file."""
        buffer = []
        with h5py.File(buffer_path, 'r') as f:
            # Assuming all datasets have the same length
            num_transitions = f['state'].shape[0]
            
            for i in range(num_transitions):
                transition = {
                    'state': f['state'][i],
                    'action': {
                        'colour': f['action_colour'][i],
                        'selection': f['action_selection'][i],
                        'transform': f['action_transform'][i]
                    },
                    'selection_mask': f['selection_mask'][i],
                    'next_state': f['next_state'][i],
                    'colour': f['colour'][i],
                    'reward': f['reward'][i],
                    'done': f['done'][i],
                    'transition_type': f['transition_type'][i],
                    'shape_h': f['shape_h'][i],
                    'shape_w': f['shape_w'][i],
                    'num_colors_grid': f['num_colors_grid'][i],
                    'most_present_color': f['most_present_color'][i],
                    'least_present_color': f['least_present_color'][i]
                }
                
                buffer.append(transition)
        return buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def _to_tensor(self, data, dtype):
        """Convert data to tensor, handling both tensor and non-tensor inputs."""
        if torch.is_tensor(data):
            return data.clone().detach().to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        transition = self.buffer[idx]
        
        # Extract state and convert to tensor
        state = self._to_tensor(transition['state'], torch.long)
        
        # Extract actions
        action_colour = self._to_tensor(transition['action']['colour'], torch.long)
        action_selection = self._to_tensor(transition['action']['selection'], torch.long)
        action_transform = self._to_tensor(transition['action']['transform'], torch.long)
        
        # Extract targets
        colour = self._to_tensor(transition['colour'], torch.long)
        selection_mask = self._to_tensor(transition['selection_mask'], torch.float32)
        
        # Extract grid statistics
        shape_h = self._to_tensor(transition['shape_h'], torch.long)
        shape_h_target = self._to_tensor(transition.get('shape_h_target', transition['shape_h']), torch.long)
        shape_h_next = self._to_tensor(transition.get('shape_h_next', transition['shape_h']), torch.long)
        shape_w = self._to_tensor(transition['shape_w'], torch.long)
        shape_w_target = self._to_tensor(transition.get('shape_w_target', transition['shape_w']), torch.long)
        shape_w_next = self._to_tensor(transition.get('shape_w_next', transition['shape_w']), torch.long)
        num_colors_grid = self._to_tensor(transition['num_colors_grid'], torch.long)
        num_colors_grid_target = self._to_tensor(transition.get('num_colors_grid_target', transition['num_colors_grid']), torch.long)
        num_colors_grid_next = self._to_tensor(transition.get('num_colors_grid_next', transition['num_colors_grid']), torch.long)
        most_present_color = self._to_tensor(transition['most_present_color'], torch.long)
        most_present_color_target = self._to_tensor(transition.get('most_present_color_target', transition['most_present_color']), torch.long)
        most_present_color_next = self._to_tensor(transition.get('most_present_color_next', transition['most_present_color']), torch.long)
        least_present_color = self._to_tensor(transition['least_present_color'], torch.long)
        least_present_color_target = self._to_tensor(transition.get('least_present_color_target', transition['least_present_color']), torch.long)
        least_present_color_next = self._to_tensor(transition.get('least_present_color_next', transition['least_present_color']), torch.long)
        
        # Prepare sample based on mode
        sample = {
            'state': state,
            'action_colour': action_colour,
            'action_selection': action_selection,
            'action_transform': action_transform,
            'colour': colour,
            'selection_mask': selection_mask,
            'reward': self._to_tensor(transition['reward'], torch.float32),
            'done': self._to_tensor(float(transition['done']), torch.float32),
            'shape_h': shape_h,
            'shape_h_target': shape_h_target,
            'shape_h_next': shape_h_next,
            'shape_w': shape_w,
            'shape_w_target': shape_w_target,
            'shape_w_next': shape_w_next,
            'num_colors_grid': num_colors_grid,
            'most_present_color': most_present_color,
            'least_present_color': least_present_color,
            'num_colors_grid_target': num_colors_grid_target,
            'most_present_color_target': most_present_color_target,
            'least_present_color_target': least_present_color_target,
            'num_colors_grid_next': num_colors_grid_next,
            'most_present_color_next': most_present_color_next,
            'least_present_color_next': least_present_color_next

        }
        
        # Add next_state for modes that need it
        if self.mode in ['selection_color', 'end_to_end']:
            next_state = self._to_tensor(transition['next_state'], torch.float32)
            sample['next_state'] = next_state
            
        target_state = self._to_tensor(transition['target_state'], torch.long)
        
        # Add transition_type if present
        transition_type = transition.get('transition_type', None)
        sample['transition_type'] = transition_type
        sample['target_state'] = target_state
        
        return sample 