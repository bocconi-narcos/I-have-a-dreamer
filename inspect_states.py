import torch
import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from src.data.replay_buffer_dataset import ReplayBufferDataset

def inspect_states(cfg: DictConfig):
    """Inspect and visualize the actual state data from the buffer."""
    print("🔍 Inspecting Replay Buffer States")
    print("=" * 50)
    
    # Buffer setup
    buffer_path = cfg.data.buffer_path
    fast_buffer_path = buffer_path
    
    # State shape
    encoder_params = OmegaConf.to_container(cfg.model.encoder.encoder_params, resolve=True)
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    print(f"📊 Buffer Path: {fast_buffer_path}")
    print(f"📐 State Shape: {state_shape}")
    print(f"🎨 Number of Colors: {cfg.num_arc_colors}")
    print()
    
    # Load dataset
    dataset = ReplayBufferDataset(
        buffer_path=fast_buffer_path,
        num_color_selection_fns=cfg.model.predictors.action_embedders.action_color_embedder.num_actions,
        num_selection_fns=cfg.model.predictors.action_embedders.action_selection_embedder.num_actions,
        num_transform_actions=cfg.model.predictors.action_embedders.action_transform_embedder.num_actions,
        num_arc_colors=cfg.num_arc_colors,
        state_shape=state_shape,
        mode='end_to_end'
    )
    
    print(f"📦 Dataset Size: {len(dataset):,} samples")
    print()
    
    # Inspect first few samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"🔍 Sample {i+1}:")
        print(f"  State shape: {sample['state'].shape}")
        print(f"  State dtype: {sample['state'].dtype}")
        print(f"  State min/max: {sample['state'].min()}/{sample['state'].max()}")
        print(f"  Grid shape: {sample['shape_h'].item()}x{sample['shape_w'].item()}")
        print(f"  Colors in grid: {sample['num_colors_grid'].item()}")
        print(f"  Most common color: {sample['most_present_color'].item()}")
        print(f"  Least common color: {sample['least_present_color'].item()}")
        print(f"  Reward: {sample['reward'].item():.2f}")
        print()
        
        # Visualize the state
        state = sample['state'].numpy()
        if state.ndim == 3 and state.shape[0] == 1:
            state = state.squeeze(0)  # Remove channel dimension
        
        plt.figure(figsize=(12, 4))
        
        # Current state
        plt.subplot(1, 3, 1)
        plt.imshow(state, cmap='tab10', vmin=-1, vmax=10)
        plt.title(f'Current State\nShape: {state.shape}')
        plt.colorbar(label='Color Index')
        
        # Next state
        next_state = sample['next_state'].numpy()
        if next_state.ndim == 3 and next_state.shape[0] == 1:
            next_state = next_state.squeeze(0)
        plt.subplot(1, 3, 2)
        plt.imshow(next_state, cmap='tab10', vmin=-1, vmax=10)
        plt.title(f'Next State\nShape: {next_state.shape}')
        plt.colorbar(label='Color Index')
        
        # Target state
        target_state = sample['target_state'].numpy()
        if target_state.ndim == 3 and target_state.shape[0] == 1:
            target_state = target_state.squeeze(0)
        plt.subplot(1, 3, 3)
        plt.imshow(target_state, cmap='tab10', vmin=-1, vmax=10)
        plt.title(f'Target State\nShape: {target_state.shape}')
        plt.colorbar(label='Color Index')
        
        plt.tight_layout()
        plt.savefig(f'state_inspection_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved visualization as 'state_inspection_sample_{i+1}.png'")
        print()
    
    # Show color distribution
    print("📊 Color Distribution Analysis:")
    print("-" * 30)
    
    colors = []
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        state = sample['state'].numpy()
        if state.ndim == 3 and state.shape[0] == 1:
            state = state.squeeze(0)
        # Flatten and get unique colors (excluding padding -1)
        unique_colors = np.unique(state[state != -1])
        colors.extend(unique_colors.tolist())
    
    unique_colors, counts = np.unique(colors, return_counts=True)
    print(f"Colors found: {unique_colors}")
    print(f"Color counts: {counts}")
    print(f"Total unique colors across samples: {len(unique_colors)}")
    print()
    
    # Show action distribution
    print("🎯 Action Distribution Analysis:")
    print("-" * 30)
    
    color_actions = []
    selection_actions = []
    transform_actions = []
    
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        color_actions.append(sample['action_colour'].item())
        selection_actions.append(sample['action_selection'].item())
        transform_actions.append(sample['action_transform'].item())
    
    print(f"Color actions range: {min(color_actions)} to {max(color_actions)}")
    print(f"Selection actions range: {min(selection_actions)} to {max(selection_actions)}")
    print(f"Transform actions range: {min(transform_actions)} to {max(transform_actions)}")
    print()
    
    # Show reward distribution
    print("💰 Reward Distribution Analysis:")
    print("-" * 30)
    
    rewards = []
    for i in range(min(1000, len(dataset))):
        sample = dataset[i]
        rewards.append(sample['reward'].item())
    
    rewards = np.array(rewards)
    print(f"Reward range: {rewards.min():.2f} to {rewards.max():.2f}")
    print(f"Reward mean: {rewards.mean():.2f}")
    print(f"Reward std: {rewards.std():.2f}")
    print(f"Positive rewards: {np.sum(rewards > 0)}/{len(rewards)} ({100*np.mean(rewards > 0):.1f}%)")
    print(f"Negative rewards: {np.sum(rewards < 0)}/{len(rewards)} ({100*np.mean(rewards < 0):.1f}%)")
    print()
    
    print("✅ State inspection complete!")

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="config")
    inspect_states(cfg) 