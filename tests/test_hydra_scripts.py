"""
Comprehensive tests for all Hydra-migrated training scripts.

This test suite verifies that:
1. All scripts can load their configs correctly
2. Config structures match expected patterns
3. Scripts initialize Hydra properly
4. Config values are accessible
5. Command-line overrides work
"""

import pytest
import sys
from pathlib import Path
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, DictConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHydraConfigs:
    """Test that all Hydra configs load correctly."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_main_config_loads(self):
        """Test that main config.yaml loads correctly."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        assert cfg is not None
        assert hasattr(cfg, 'data')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'training')
        assert hasattr(cfg, 'latent_dim')
    
    def test_autoencoder_config_loads(self):
        """Test that config_autoencoder.yaml loads correctly."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config_autoencoder")
        
        assert cfg is not None
        assert hasattr(cfg, 'data')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'training')
        assert hasattr(cfg, 'latent_dim')
        assert hasattr(cfg.data, 'autoencoder')
    
    def test_main_config_structure(self):
        """Test main config has expected structure."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Data config
        assert hasattr(cfg.data, 'buffer_path')
        
        # Model config
        assert hasattr(cfg.model, 'encoder')
        assert hasattr(cfg.model, 'predictors')
        assert hasattr(cfg.model.encoder, 'encoder_params')
        
        # Training config
        assert hasattr(cfg.training, 'batch_size')
        assert hasattr(cfg.training, 'learning_rate')
        assert hasattr(cfg.training, 'num_epochs')
    
    def test_autoencoder_config_structure(self):
        """Test autoencoder config has expected structure."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config_autoencoder")
        
        # Data config
        assert hasattr(cfg.data.autoencoder, 'buffer_path')
        
        # Model config
        assert hasattr(cfg.model.encoder, 'encoder_params')
        assert hasattr(cfg.model.decoder, 'decoder_params')
        
        # Training config
        assert hasattr(cfg.training.autoencoder, 'batch_size')
        assert hasattr(cfg.training.autoencoder, 'learning_rate')
        assert hasattr(cfg.training.autoencoder, 'num_epochs')
    
    def test_config_value_types(self):
        """Test that config values have correct types."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        assert isinstance(cfg.training.batch_size, (int, float))
        assert isinstance(cfg.training.learning_rate, (int, float))
        assert isinstance(cfg.latent_dim, int)
        assert isinstance(cfg.data.buffer_path, str)
    
    def test_config_overrides(self):
        """Test that command-line overrides work."""
        initialize(config_path="../conf", version_base=None)
        
        # Test override
        cfg = compose(config_name="config", overrides=["training.batch_size=128"])
        assert cfg.training.batch_size == 128
        
        # Test multiple overrides
        cfg = compose(config_name="config", overrides=[
            "training.batch_size=64",
            "training.learning_rate=0.001"
        ])
        assert cfg.training.batch_size == 64
        assert cfg.training.learning_rate == 0.001


class TestScriptImports:
    """Test that all migrated scripts can be imported and have correct structure."""
    
    def test_train_color_predictor_imports(self):
        """Test train_color_predictor.py imports correctly."""
        import train_color_predictor
        assert hasattr(train_color_predictor, 'train_color_predictor')
        assert hasattr(train_color_predictor, 'compose')
        assert hasattr(train_color_predictor, 'initialize')
    
    def test_train_autoencoder_imports(self):
        """Test train_autoencoder.py imports correctly."""
        import train_autoencoder
        assert hasattr(train_autoencoder, 'train_autoencoder')
        assert hasattr(train_autoencoder, 'compose')
        assert hasattr(train_autoencoder, 'initialize')
    
    def test_train_step_distance_encoder_imports(self):
        """Test train_step_distance_encoder.py imports correctly."""
        import train_step_distance_encoder
        assert hasattr(train_step_distance_encoder, 'train_step_distance_encoder')
        assert hasattr(train_step_distance_encoder, 'compose')
        assert hasattr(train_step_distance_encoder, 'initialize')
    
    def test_train_selection_predictor_imports(self):
        """Test train_selection_predictor.py imports correctly."""
        import train_selection_predictor
        assert hasattr(train_selection_predictor, 'train_selection_predictor')
        assert hasattr(train_selection_predictor, 'compose')
        assert hasattr(train_selection_predictor, 'initialize')
    
    def test_train_next_state_predictor_imports(self):
        """Test train_next_state_predictor.py imports correctly."""
        import train_next_state_predictor
        assert hasattr(train_next_state_predictor, 'train_next_state_predictor')
        assert hasattr(train_next_state_predictor, 'compose')
        assert hasattr(train_next_state_predictor, 'initialize')
    
    def test_train_reward_predictor_imports(self):
        """Test train_reward_predictor.py imports correctly."""
        import train_reward_predictor
        assert hasattr(train_reward_predictor, 'train_reward_predictor')
        assert hasattr(train_reward_predictor, 'compose')
        assert hasattr(train_reward_predictor, 'initialize')
    
    def test_train_full_model_imports(self):
        """Test train_full_model.py imports correctly."""
        import train_full_model
        assert hasattr(train_full_model, 'train_full_model')
        assert hasattr(train_full_model, 'compose')
        assert hasattr(train_full_model, 'initialize')
    
    def test_train_step_distance_mlp_imports(self):
        """Test train_step_distance_mlp.py imports correctly."""
        import train_step_distance_mlp
        assert hasattr(train_step_distance_mlp, 'train_step_distance_mlp')
        assert hasattr(train_step_distance_mlp, 'compose')
        assert hasattr(train_step_distance_mlp, 'initialize')


class TestScriptConfigLoading:
    """Test that scripts can load their configs correctly."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_train_color_predictor_config(self):
        """Test train_color_predictor loads main config."""
        import train_color_predictor
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure matches what script expects
        assert hasattr(cfg, 'data')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'training')
        assert hasattr(cfg.data, 'buffer_path')
        assert hasattr(cfg.model.encoder, 'encoder_params')
        assert hasattr(cfg.training, 'batch_size')
    
    def test_train_autoencoder_config(self):
        """Test train_autoencoder loads autoencoder config."""
        import train_autoencoder
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config_autoencoder")
        
        # Verify config structure matches what script expects
        assert hasattr(cfg, 'data')
        assert hasattr(cfg, 'model')
        assert hasattr(cfg, 'training')
        assert hasattr(cfg.data.autoencoder, 'buffer_path')
        assert hasattr(cfg.model.encoder, 'encoder_params')
        assert hasattr(cfg.model.decoder, 'decoder_params')
        assert hasattr(cfg.training.autoencoder, 'batch_size')
    
    def test_train_step_distance_encoder_config(self):
        """Test train_step_distance_encoder loads autoencoder config."""
        import train_step_distance_encoder
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config_autoencoder")
        
        # Verify config structure
        assert hasattr(cfg.data.autoencoder, 'buffer_path')
        assert hasattr(cfg.model.encoder, 'encoder_params')
        assert hasattr(cfg.training.autoencoder, 'batch_size')
    
    def test_train_selection_predictor_config(self):
        """Test train_selection_predictor loads main config."""
        import train_selection_predictor
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure
        assert hasattr(cfg.model.predictors, 'selection_mask')
        assert hasattr(cfg.model.predictors.selection_mask, 'mask_encoder_params')
        assert hasattr(cfg.model.predictors.selection_mask, 'mask_predictor_params')
    
    def test_train_next_state_predictor_config(self):
        """Test train_next_state_predictor loads main config."""
        import train_next_state_predictor
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure
        assert hasattr(cfg.model.predictors, 'next_state')
        assert hasattr(cfg.model.predictors.next_state, 'transformer_depth')
        assert hasattr(cfg.model.predictors, 'selection_mask')
    
    def test_train_reward_predictor_config(self):
        """Test train_reward_predictor loads main config."""
        import train_reward_predictor
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure
        assert hasattr(cfg.model.predictors, 'reward_predictor')
        assert hasattr(cfg.model.predictors.reward_predictor, 'hidden_dim')
    
    def test_train_full_model_config(self):
        """Test train_full_model loads main config."""
        import train_full_model
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure
        assert hasattr(cfg.model.predictors, 'reward_predictor')
        assert hasattr(cfg.model.predictors, 'continuation_predictor')
        assert hasattr(cfg.model.predictors, 'next_state')
    
    def test_train_step_distance_mlp_config(self):
        """Test train_step_distance_mlp loads main config."""
        import train_step_distance_mlp
        from hydra import compose, initialize
        
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Verify config structure
        assert hasattr(cfg.data, 'buffer_path')
        assert hasattr(cfg.model.encoder, 'encoder_params')
        assert hasattr(cfg.training, 'batch_size')


class TestConfigAccessPatterns:
    """Test that config access patterns match script expectations."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_omega_conf_to_container(self):
        """Test OmegaConf.to_container works for nested configs."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Test converting encoder_params
        encoder_params = OmegaConf.to_container(cfg.model.encoder.encoder_params, resolve=True)
        assert isinstance(encoder_params, dict)
        assert 'depth' in encoder_params or 'image_size' in encoder_params
        
        # Test converting predictor params
        if hasattr(cfg.model.predictors, 'selection_mask'):
            mask_params = OmegaConf.to_container(
                cfg.model.predictors.selection_mask.mask_encoder_params, 
                resolve=True
            )
            assert isinstance(mask_params, dict)
    
    def test_omega_conf_select(self):
        """Test OmegaConf.select works for optional values."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Test selecting existing value
        batch_size = OmegaConf.select(cfg.training, 'batch_size', default=32)
        assert batch_size is not None
        
        # Test selecting non-existent value with default
        non_existent = OmegaConf.select(cfg, 'non_existent_key', default='default_value')
        assert non_existent == 'default_value'
    
    def test_nested_config_access(self):
        """Test nested config access works correctly."""
        initialize(config_path="../conf", version_base=None)
        cfg = compose(config_name="config")
        
        # Test deeply nested access
        if hasattr(cfg.model.predictors, 'action_embedders'):
            if hasattr(cfg.model.predictors.action_embedders, 'action_color_embedder'):
                num_actions = cfg.model.predictors.action_embedders.action_color_embedder.num_actions
                assert isinstance(num_actions, (int, float))


class TestConfigConsistency:
    """Test config consistency across scripts."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_latent_dim_consistency(self):
        """Test that latent_dim is consistent across configs."""
        initialize(config_path="../conf", version_base=None)
        
        cfg_main = compose(config_name="config")
        cfg_ae = compose(config_name="config_autoencoder")
        
        # Both should have latent_dim
        assert hasattr(cfg_main, 'latent_dim')
        assert hasattr(cfg_ae, 'latent_dim')
        
        # They might have different values, but both should be integers
        assert isinstance(cfg_main.latent_dim, int)
        assert isinstance(cfg_ae.latent_dim, int)
    
    def test_encoder_params_structure(self):
        """Test that encoder_params have consistent structure."""
        initialize(config_path="../conf", version_base=None)
        
        cfg_main = compose(config_name="config")
        cfg_ae = compose(config_name="config_autoencoder")
        
        # Both should have encoder_params
        main_encoder = OmegaConf.to_container(cfg_main.model.encoder.encoder_params, resolve=True)
        ae_encoder = OmegaConf.to_container(cfg_ae.model.encoder.encoder_params, resolve=True)
        
        assert isinstance(main_encoder, dict)
        assert isinstance(ae_encoder, dict)
        
        # Both should have common keys
        common_keys = set(main_encoder.keys()) & set(ae_encoder.keys())
        assert len(common_keys) > 0  # Should share some common parameters


class TestScriptInitialization:
    """Test that scripts initialize Hydra correctly."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_global_hydra_check(self):
        """Test GlobalHydra().is_initialized() works."""
        assert not GlobalHydra().is_initialized()
        
        initialize(config_path="../conf", version_base=None)
        assert GlobalHydra().is_initialized()
        
        GlobalHydra.instance().clear()
        assert not GlobalHydra().is_initialized()
    
    def test_script_initialization_pattern(self):
        """Test that scripts follow correct initialization pattern."""
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        
        # Simulate script initialization
        if not GlobalHydra().is_initialized():
            initialize(config_path="../conf", version_base=None)
        
        cfg = compose(config_name="config")
        assert cfg is not None
        
        # Cleanup
        GlobalHydra.instance().clear()


class TestConfigValidation:
    """Test config validation and error handling."""
    
    def setup_method(self):
        """Reset Hydra before each test."""
        GlobalHydra.instance().clear()
    
    def test_invalid_config_name_raises_error(self):
        """Test that invalid config name raises error."""
        initialize(config_path="../conf", version_base=None)
        
        with pytest.raises(Exception):  # Should raise MissingConfigException or similar
            compose(config_name="nonexistent_config")
    
    def test_invalid_override_raises_error(self):
        """Test that invalid override raises error."""
        initialize(config_path="../conf", version_base=None)
        
        with pytest.raises(Exception):  # Should raise ConfigCompositionException or similar
            compose(config_name="config", overrides=["nonexistent.key=value"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

