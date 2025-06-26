import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from torch.utils.data import DataLoader
import os

from src.models.state_encoder import StateEncoder
from src.models.color_predictor import ColorPredictor
from src.models.mask_encoder import MaskEncoder
from src.models.selection_mask_predictor import SelectionMaskPredictor
from src.models.next_state_predictor import NextStatePredictor
from src.losses.vicreg import VICRegLoss
from src.data import ReplayBufferDataset
from src.models.reward_predictor import RewardPredictor
from src.models.continuation_predictor import ContinuationPredictor

def load_config(config_path="unified_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_dimensions_and_shapes():
    """Test that all model dimensions and tensor shapes are correct."""
    print("=" * 60)
    print("TESTING DIMENSIONS AND SHAPES")
    print("=" * 60)
    
    config = load_config()
    
    # Extract config parameters
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params'][encoder_type]
    num_color_selection_fns = config['num_color_selection_fns']
    num_selection_fns = config['num_selection_fns']
    num_transform_actions = config['num_transform_actions']
    num_arc_colors = config['num_arc_colors']
    color_predictor_hidden_dim = config['color_predictor']['hidden_dim']
    
    # Selection mask config
    selection_cfg = config['selection_mask']
    mask_encoder_type = selection_cfg['mask_encoder_type']
    mask_encoder_params = selection_cfg['mask_encoder_params'][mask_encoder_type]
    latent_mask_dim = selection_cfg['latent_mask_dim']
    selection_mask_predictor_hidden_dim = selection_cfg['selection_mask_predictor_hidden_dim']
    use_transformer_selection = selection_cfg['use_transformer']
    transformer_depth_selection = selection_cfg['transformer_depth']
    transformer_heads_selection = selection_cfg['transformer_heads']
    transformer_dim_head_selection = selection_cfg['transformer_dim_head']
    transformer_mlp_dim_selection = selection_cfg['transformer_mlp_dim']
    transformer_dropout_selection = selection_cfg['transformer_dropout']
    
    # Next state config
    next_state_cfg = config['next_state']
    latent_mask_dim_next_state = next_state_cfg['latent_mask_dim']
    transformer_depth_next_state = next_state_cfg['transformer_depth']
    transformer_heads_next_state = next_state_cfg['transformer_heads']
    transformer_dim_head_next_state = next_state_cfg['transformer_dim_head']
    transformer_mlp_dim_next_state = next_state_cfg['transformer_mlp_dim']
    transformer_dropout_next_state = next_state_cfg['transformer_dropout']
    
    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    print(f"State shape: {state_shape}")
    print(f"Latent dim: {latent_dim}")
    print(f"Latent mask dim: {latent_mask_dim}")
    print(f"Num color selection functions: {num_color_selection_fns}")
    print(f"Num selection functions: {num_selection_fns}")
    print(f"Num transform actions: {num_transform_actions}")
    print(f"Num ARC colors: {num_arc_colors}")
    
    # Test batch size
    batch_size = 4
    
    # Create dummy data
    state = torch.randn(batch_size, *state_shape)
    next_state = torch.randn(batch_size, *state_shape)
    selection_mask = torch.randint(0, 2, (batch_size, *state_shape)).float()
    
    action_colour = torch.randint(0, num_color_selection_fns, (batch_size,))
    action_selection = torch.randint(0, num_selection_fns, (batch_size,))
    action_transform = torch.randint(0, num_transform_actions, (batch_size,))
    target_colour = torch.randint(0, num_arc_colors, (batch_size,))
    
    print(f"\nInput shapes:")
    print(f"State: {state.shape}")
    print(f"Next state: {next_state.shape}")
    print(f"Selection mask: {selection_mask.shape}")
    print(f"Action colour: {action_colour.shape}")
    print(f"Action selection: {action_selection.shape}")
    print(f"Action transform: {action_transform.shape}")
    print(f"Target colour: {target_colour.shape}")
    
    # Initialize models
    device = torch.device('cpu')
    
    state_encoder = StateEncoder(encoder_type, latent_dim=latent_dim, **encoder_params).to(device)
    color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)
    mask_encoder = MaskEncoder(mask_encoder_type, latent_dim=latent_mask_dim, **mask_encoder_params).to(device)
    selection_mask_predictor = SelectionMaskPredictor(
        input_dim=latent_dim + num_selection_fns,
        latent_mask_dim=latent_mask_dim,
        hidden_dim=selection_mask_predictor_hidden_dim,
        use_transformer=use_transformer_selection,
        transformer_depth=transformer_depth_selection,
        transformer_heads=transformer_heads_selection,
        transformer_dim_head=transformer_dim_head_selection,
        transformer_mlp_dim=transformer_mlp_dim_selection,
        dropout=transformer_dropout_selection
    ).to(device)
    next_state_predictor = NextStatePredictor(
        latent_dim=latent_dim,
        num_transform_actions=num_transform_actions,
        latent_mask_dim=latent_mask_dim_next_state,
        transformer_depth=transformer_depth_next_state,
        transformer_heads=transformer_heads_next_state,
        transformer_dim_head=transformer_dim_head_next_state,
        transformer_mlp_dim=transformer_mlp_dim_next_state,
        dropout=transformer_dropout_next_state
    ).to(device)
    
    # Test forward passes
    print(f"\nTesting forward passes:")
    
    # State encoding
    latent = state_encoder(state.float())
    print(f"State encoder output: {latent.shape}")
    assert latent.shape == (batch_size, latent_dim), f"Expected {(batch_size, latent_dim)}, got {latent.shape}"
    
    # Color prediction
    action_colour_onehot = torch.nn.functional.one_hot(action_colour, num_color_selection_fns).float()
    color_input = torch.cat([latent, action_colour_onehot], dim=1)
    color_logits = color_predictor(color_input)
    print(f"Color predictor output: {color_logits.shape}")
    assert color_logits.shape == (batch_size, num_arc_colors), f"Expected {(batch_size, num_arc_colors)}, got {color_logits.shape}"
    
    # Selection mask prediction
    action_selection_onehot = torch.nn.functional.one_hot(action_selection, num_selection_fns).float()
    selection_input = torch.cat([latent, action_selection_onehot], dim=1)
    pred_latent_mask = selection_mask_predictor(selection_input)
    print(f"Selection mask predictor output: {pred_latent_mask.shape}")
    assert pred_latent_mask.shape == (batch_size, latent_mask_dim), f"Expected {(batch_size, latent_mask_dim)}, got {pred_latent_mask.shape}"
    
    # Mask encoding
    target_latent_mask = mask_encoder(selection_mask.float())
    print(f"Mask encoder output: {target_latent_mask.shape}")
    assert target_latent_mask.shape == (batch_size, latent_mask_dim), f"Expected {(batch_size, latent_mask_dim)}, got {target_latent_mask.shape}"
    
    # Next state prediction
    action_transform_onehot = torch.nn.functional.one_hot(action_transform, num_transform_actions).float()
    latent_next = state_encoder(next_state.float())
    pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
    print(f"Next state predictor output: {pred_next_latent.shape}")
    assert pred_next_latent.shape == (batch_size, latent_dim), f"Expected {(batch_size, latent_dim)}, got {pred_next_latent.shape}"
    
    # --- Reward Predictor and Continuation Predictor input shape tests ---
    # Transformer mode
    reward_predictor = RewardPredictor(input_dim=latent_dim, use_transformer=True)
    continuation_predictor = ContinuationPredictor(input_dim=latent_dim, use_transformer=True)
    stacked = torch.stack([latent, pred_next_latent], dim=1)  # (B, 2, latent_dim)
    reward_out = reward_predictor(stacked)
    continuation_out = continuation_predictor(stacked)
    print(f"Reward predictor (transformer) output: {reward_out.shape}")
    print(f"Continuation predictor (transformer) output: {continuation_out.shape}")
    assert reward_out.shape == (batch_size,), f"Expected {(batch_size,)}, got {reward_out.shape}"
    assert continuation_out.shape == (batch_size,), f"Expected {(batch_size,)}, got {continuation_out.shape}"

    # MLP mode
    reward_predictor_mlp = RewardPredictor(input_dim=latent_dim, use_transformer=False)
    continuation_predictor_mlp = ContinuationPredictor(input_dim=latent_dim, use_transformer=False)
    concatenated = torch.cat([latent, pred_next_latent], dim=1)  # (B, 2*latent_dim)
    reward_out_mlp = reward_predictor_mlp(concatenated)
    continuation_out_mlp = continuation_predictor_mlp(concatenated)
    print(f"Reward predictor (MLP) output: {reward_out_mlp.shape}")
    print(f"Continuation predictor (MLP) output: {continuation_out_mlp.shape}")
    assert reward_out_mlp.shape == (batch_size,), f"Expected {(batch_size,)}, got {reward_out_mlp.shape}"
    assert continuation_out_mlp.shape == (batch_size,), f"Expected {(batch_size,)}, got {continuation_out_mlp.shape}"
    
    print("✅ All dimension tests passed!")

def test_replay_buffer():
    """Test the unified replay buffer with different modes."""
    print("\n" + "=" * 60)
    print("TESTING REPLAY BUFFER")
    print("=" * 60)
    
    config = load_config()
    
    # Extract config parameters
    buffer_path = config['buffer_path']
    encoder_params = config['encoder_params'][config['encoder_type']]
    num_color_selection_fns = config['num_color_selection_fns']
    num_selection_fns = config['num_selection_fns']
    num_transform_actions = config['num_transform_actions']
    num_arc_colors = config['num_arc_colors']
    
    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    # Test all modes
    modes = ['color_only', 'selection_color', 'end_to_end']
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        try:
            dataset = ReplayBufferDataset(
                buffer_path=buffer_path,
                num_color_selection_fns=num_color_selection_fns,
                num_selection_fns=num_selection_fns,
                num_transform_actions=num_transform_actions,
                num_arc_colors=num_arc_colors,
                state_shape=state_shape,
                mode=mode,
                num_samples=10  # Small sample for testing
            )
            
            print(f"Dataset length: {len(dataset)}")
            
            # Test a few samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Sample {i} keys: {list(sample.keys())}")
                
                # Check required fields
                assert 'state' in sample, f"Missing 'state' in {mode}"
                assert 'action_colour' in sample, f"Missing 'action_colour' in {mode}"
                assert 'action_selection' in sample, f"Missing 'action_selection' in {mode}"
                assert 'action_transform' in sample, f"Missing 'action_transform' in {mode}"
                assert 'colour' in sample, f"Missing 'colour' in {mode}"
                assert 'selection_mask' in sample, f"Missing 'selection_mask' in {mode}"
                
                # Check mode-specific fields
                if mode in ['selection_color', 'end_to_end']:
                    assert 'next_state' in sample, f"Missing 'next_state' in {mode}"
                
                # Check shapes
                assert sample['state'].shape == state_shape, f"Wrong state shape in {mode}"
                assert sample['selection_mask'].shape == state_shape, f"Wrong selection_mask shape in {mode}"
                if mode in ['selection_color', 'end_to_end']:
                    assert sample['next_state'].shape == state_shape, f"Wrong next_state shape in {mode}"
                
                # Check value ranges
                assert 0 <= sample['action_colour'].item() < num_color_selection_fns, f"Invalid action_colour in {mode}"
                assert 0 <= sample['action_selection'].item() < num_selection_fns, f"Invalid action_selection in {mode}"
                assert 0 <= sample['action_transform'].item() < num_transform_actions, f"Invalid action_transform in {mode}"
                assert 0 <= sample['colour'].item() < num_arc_colors, f"Invalid colour in {mode}"
            
            print(f"✅ {mode} mode test passed!")
            
        except Exception as e:
            print(f"❌ {mode} mode test failed: {e}")
            raise

def test_training_loops():
    """Test that training loops can run without errors."""
    print("\n" + "=" * 60)
    print("TESTING TRAINING LOOPS")
    print("=" * 60)
    
    config = load_config()
    
    # Extract config parameters
    buffer_path = config['buffer_path']
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params'][encoder_type]
    num_color_selection_fns = config['num_color_selection_fns']
    num_selection_fns = config['num_selection_fns']
    num_transform_actions = config['num_transform_actions']
    num_arc_colors = config['num_arc_colors']
    color_predictor_hidden_dim = config['color_predictor']['hidden_dim']
    
    # Selection mask config
    selection_cfg = config['selection_mask']
    mask_encoder_type = selection_cfg['mask_encoder_type']
    mask_encoder_params = selection_cfg['mask_encoder_params'][mask_encoder_type]
    latent_mask_dim = selection_cfg['latent_mask_dim']
    selection_mask_predictor_hidden_dim = selection_cfg['selection_mask_predictor_hidden_dim']
    use_transformer_selection = selection_cfg['use_transformer']
    transformer_depth_selection = selection_cfg['transformer_depth']
    transformer_heads_selection = selection_cfg['transformer_heads']
    transformer_dim_head_selection = selection_cfg['transformer_dim_head']
    transformer_mlp_dim_selection = selection_cfg['transformer_mlp_dim']
    transformer_dropout_selection = selection_cfg['transformer_dropout']
    use_vicreg_selection = selection_cfg['use_vicreg']
    vicreg_sim_coeff_selection = selection_cfg['vicreg_sim_coeff']
    vicreg_std_coeff_selection = selection_cfg['vicreg_std_coeff']
    vicreg_cov_coeff_selection = selection_cfg['vicreg_cov_coeff']
    
    # Next state config
    next_state_cfg = config['next_state']
    latent_mask_dim_next_state = next_state_cfg['latent_mask_dim']
    transformer_depth_next_state = next_state_cfg['transformer_depth']
    transformer_heads_next_state = next_state_cfg['transformer_heads']
    transformer_dim_head_next_state = next_state_cfg['transformer_dim_head']
    transformer_mlp_dim_next_state = next_state_cfg['transformer_mlp_dim']
    transformer_dropout_next_state = next_state_cfg['transformer_dropout']
    use_vicreg_next_state = next_state_cfg['use_vicreg']
    vicreg_sim_coeff_next_state = next_state_cfg['vicreg_sim_coeff']
    vicreg_std_coeff_next_state = next_state_cfg['vicreg_std_coeff']
    vicreg_cov_coeff_next_state = next_state_cfg['vicreg_cov_coeff']
    
    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    device = torch.device('cpu')
    batch_size = 2
    num_epochs = 1
    
    # Test 1: Color predictor only
    print("\nTesting color predictor training loop...")
    try:
        dataset = ReplayBufferDataset(
            buffer_path=buffer_path,
            num_color_selection_fns=num_color_selection_fns,
            num_selection_fns=num_selection_fns,
            num_transform_actions=num_transform_actions,
            num_arc_colors=num_arc_colors,
            state_shape=state_shape,
            mode='color_only',
            num_samples=10
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        state_encoder = StateEncoder(encoder_type, latent_dim=latent_dim, **encoder_params).to(device)
        color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(list(state_encoder.parameters()) + list(color_predictor.parameters()), lr=0.001)
        
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                state = batch['state'].to(device)
                action_colour = batch['action_colour'].to(device)
                target_colour = batch['colour'].to(device)
                
                if state.dim() == 3:
                    state = state.unsqueeze(1)
                
                latent = state_encoder(state.float())
                action_colour_onehot = torch.nn.functional.one_hot(action_colour, num_color_selection_fns).float()
                x = torch.cat([latent, action_colour_onehot], dim=1)
                logits = color_predictor(x)
                
                loss = criterion(logits, target_colour)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i == 0:
                    print(f"Color training - Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")
                    break
        
        print("✅ Color predictor training loop test passed!")
        
    except Exception as e:
        print(f"❌ Color predictor training loop test failed: {e}")
        raise
    
    # Test 2: Selection + Color training
    print("\nTesting selection + color training loop...")
    try:
        dataset = ReplayBufferDataset(
            buffer_path=buffer_path,
            num_color_selection_fns=num_color_selection_fns,
            num_selection_fns=num_selection_fns,
            num_transform_actions=num_transform_actions,
            num_arc_colors=num_arc_colors,
            state_shape=state_shape,
            mode='selection_color',
            num_samples=10
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        mask_encoder = MaskEncoder(mask_encoder_type, latent_dim=latent_mask_dim, **mask_encoder_params).to(device)
        selection_mask_predictor = SelectionMaskPredictor(
            input_dim=latent_dim + num_selection_fns,
            latent_mask_dim=latent_mask_dim,
            hidden_dim=selection_mask_predictor_hidden_dim,
            use_transformer=use_transformer_selection,
            transformer_depth=transformer_depth_selection,
            transformer_heads=transformer_heads_selection,
            transformer_dim_head=transformer_dim_head_selection,
            transformer_mlp_dim=transformer_mlp_dim_selection,
            dropout=transformer_dropout_selection
        ).to(device)
        
        color_criterion = nn.CrossEntropyLoss()
        selection_criterion = nn.MSELoss()
        vicreg_loss_fn = VICRegLoss(sim_coeff=vicreg_sim_coeff_selection, std_coeff=vicreg_std_coeff_selection, cov_coeff=vicreg_cov_coeff_selection)
        
        optimizer = optim.AdamW(
            list(state_encoder.parameters()) + 
            list(color_predictor.parameters()) + 
            list(mask_encoder.parameters()) + 
            list(selection_mask_predictor.parameters()), 
            lr=0.001
        )
        
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                state = batch['state'].to(device)
                action_colour = batch['action_colour'].to(device)
                action_selection = batch['action_selection'].to(device)
                target_colour = batch['colour'].to(device)
                selection_mask = batch['selection_mask'].to(device)
                
                if state.dim() == 3:
                    state = state.unsqueeze(1)
                    selection_mask = selection_mask.unsqueeze(1)
                
                latent = state_encoder(state.float())
                action_colour_onehot = torch.nn.functional.one_hot(action_colour, num_color_selection_fns).float()
                action_selection_onehot = torch.nn.functional.one_hot(action_selection, num_selection_fns).float()
                
                # Color prediction
                color_input = torch.cat([latent, action_colour_onehot], dim=1)
                color_logits = color_predictor(color_input)
                color_loss = color_criterion(color_logits, target_colour)
                
                # Selection mask prediction
                selection_input = torch.cat([latent, action_selection_onehot], dim=1)
                pred_latent_mask = selection_mask_predictor(selection_input)
                target_latent_mask = mask_encoder(selection_mask.float())
                
                if use_vicreg_selection:
                    selection_loss, _, _, _ = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
                else:
                    selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)
                
                total_loss = color_loss + selection_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if i == 0:
                    print(f"Selection+Color training - Epoch {epoch+1}, Batch {i+1}, Color Loss: {color_loss.item():.4f}, Selection Loss: {selection_loss.item():.4f}")
                    break
        
        print("✅ Selection + Color training loop test passed!")
        
    except Exception as e:
        print(f"❌ Selection + Color training loop test failed: {e}")
        raise
    
    # Test 3: End-to-end training
    print("\nTesting end-to-end training loop...")
    try:
        dataset = ReplayBufferDataset(
            buffer_path=buffer_path,
            num_color_selection_fns=num_color_selection_fns,
            num_selection_fns=num_selection_fns,
            num_transform_actions=num_transform_actions,
            num_arc_colors=num_arc_colors,
            state_shape=state_shape,
            mode='end_to_end',
            num_samples=10
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        next_state_predictor = NextStatePredictor(
            latent_dim=latent_dim,
            num_transform_actions=num_transform_actions,
            latent_mask_dim=latent_mask_dim_next_state,
            transformer_depth=transformer_depth_next_state,
            transformer_heads=transformer_heads_next_state,
            transformer_dim_head=transformer_dim_head_next_state,
            transformer_mlp_dim=transformer_mlp_dim_next_state,
            dropout=transformer_dropout_next_state
        ).to(device)
        
        next_state_criterion = nn.MSELoss()
        vicreg_loss_fn_next_state = VICRegLoss(sim_coeff=vicreg_sim_coeff_next_state, std_coeff=vicreg_std_coeff_next_state, cov_coeff=vicreg_cov_coeff_next_state)
        
        optimizer = optim.AdamW(
            list(state_encoder.parameters()) + 
            list(color_predictor.parameters()) + 
            list(mask_encoder.parameters()) + 
            list(selection_mask_predictor.parameters()) + 
            list(next_state_predictor.parameters()), 
            lr=0.001
        )
        
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                state = batch['state'].to(device)
                next_state = batch['next_state'].to(device)
                action_colour = batch['action_colour'].to(device)
                action_selection = batch['action_selection'].to(device)
                action_transform = batch['action_transform'].to(device)
                target_colour = batch['colour'].to(device)
                selection_mask = batch['selection_mask'].to(device)
                
                if state.dim() == 3:
                    state = state.unsqueeze(1)
                    next_state = next_state.unsqueeze(1)
                    selection_mask = selection_mask.unsqueeze(1)
                
                latent = state_encoder(state.float())
                latent_next = state_encoder(next_state.float())
                action_colour_onehot = torch.nn.functional.one_hot(action_colour, num_color_selection_fns).float()
                action_selection_onehot = torch.nn.functional.one_hot(action_selection, num_selection_fns).float()
                action_transform_onehot = torch.nn.functional.one_hot(action_transform, num_transform_actions).float()
                
                # Color prediction
                color_input = torch.cat([latent, action_colour_onehot], dim=1)
                color_logits = color_predictor(color_input)
                color_loss = color_criterion(color_logits, target_colour)
                
                # Selection mask prediction
                selection_input = torch.cat([latent, action_selection_onehot], dim=1)
                pred_latent_mask = selection_mask_predictor(selection_input)
                target_latent_mask = mask_encoder(selection_mask.float())
                
                if use_vicreg_selection:
                    selection_loss, _, _, _ = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
                else:
                    selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)
                
                # Next state prediction
                pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
                
                if use_vicreg_next_state:
                    next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
                else:
                    next_state_loss = next_state_criterion(pred_next_latent, latent_next)
                
                total_loss = color_loss + selection_loss + next_state_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if i == 0:
                    print(f"End-to-end training - Epoch {epoch+1}, Batch {i+1}")
                    print(f"  Color Loss: {color_loss.item():.4f}")
                    print(f"  Selection Loss: {selection_loss.item():.4f}")
                    print(f"  Next State Loss: {next_state_loss.item():.4f}")
                    print(f"  Total Loss: {total_loss.item():.4f}")
                    break
        
        print("✅ End-to-end training loop test passed!")
        
    except Exception as e:
        print(f"❌ End-to-end training loop test failed: {e}")
        raise

def test_real_replay_buffer(buffer_path):
    """Test with a real replay buffer file."""
    print("\n" + "=" * 60)
    print("TESTING REAL REPLAY BUFFER")
    print("=" * 60)
    
    if not os.path.exists(buffer_path):
        print(f"⚠️  Real replay buffer not found at {buffer_path}")
        print("   Skipping real buffer test. Create a buffer file to test this.")
        return
    
    config = load_config()
    
    # Extract config parameters
    encoder_params = config['encoder_params'][config['encoder_type']]
    num_color_selection_fns = config['num_color_selection_fns']
    num_selection_fns = config['num_selection_fns']
    num_transform_actions = config['num_transform_actions']
    num_arc_colors = config['num_arc_colors']
    
    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    try:
        print(f"Loading real replay buffer from: {buffer_path}")
        
        # Test all modes with real buffer
        modes = ['color_only', 'selection_color', 'end_to_end']
        
        for mode in modes:
            print(f"\nTesting real buffer with mode: {mode}")
            dataset = ReplayBufferDataset(
                buffer_path=buffer_path,
                num_color_selection_fns=num_color_selection_fns,
                num_selection_fns=num_selection_fns,
                num_transform_actions=num_transform_actions,
                num_arc_colors=num_arc_colors,
                state_shape=state_shape,
                mode=mode
            )
            
            print(f"Real buffer dataset length: {len(dataset)}")
            
            # Test a few samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"Real sample {i} keys: {list(sample.keys())}")
                
                # Check shapes
                print(f"  State shape: {sample['state'].shape}")
                print(f"  Selection mask shape: {sample['selection_mask'].shape}")
                if mode in ['selection_color', 'end_to_end']:
                    print(f"  Next state shape: {sample['next_state'].shape}")
                
                # Check value ranges
                print(f"  Action colour: {sample['action_colour'].item()}")
                print(f"  Action selection: {sample['action_selection'].item()}")
                print(f"  Action transform: {sample['action_transform'].item()}")
                print(f"  Target colour: {sample['colour'].item()}")
            
            print(f"✅ Real buffer {mode} mode test passed!")
        
    except Exception as e:
        print(f"❌ Real replay buffer test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE TRAINING PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Test 1: Dimensions and shapes
        test_dimensions_and_shapes()
        
        # Test 2: Replay buffer
        test_replay_buffer()
        
        # Test 3: Training loops
        test_training_loops()
        
        # Test 4: Real replay buffer (if available)
        test_real_replay_buffer("buffer.pkl")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Training pipeline is ready.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 