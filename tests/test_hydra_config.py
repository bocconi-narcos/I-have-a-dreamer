"""
Test Hydra configuration loading
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    pytest.skip("Hydra not installed", allow_module_level=True)


class TestHydraConfig:
    """Test Hydra configuration loading"""
    
    def test_config_loads(self):
        """Test that main config loads successfully"""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        # Initialize Hydra if not already initialized
        # Use relative path from project root (tests/ -> ../conf)
        if not GlobalHydra().is_initialized():
            initialize(config_path="../conf", version_base=None)
        
        # Compose config
        cfg = compose(config_name="config")
        
        # Test basic access
        assert cfg.data.buffer_path is not None
        assert cfg.latent_dim is not None
        assert cfg.training.batch_size is not None
        assert cfg.training.learning_rate is not None
    
    def test_config_structure(self):
        """Test that config has expected structure"""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        if not GlobalHydra().is_initialized():
            initialize(config_path="../conf", version_base=None)
        
        cfg = compose(config_name="config")
        
        # Test nested access
        assert cfg.model.encoder.encoder_params.depth is not None
        assert cfg.model.predictors.color_predictor.hidden_dim is not None
        assert cfg.model.predictors.action_embedders.action_color_embedder.num_actions is not None
    
    def test_config_values(self):
        """Test that config values are correct"""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available")
        
        if not GlobalHydra().is_initialized():
            initialize(config_path="../conf", version_base=None)
        
        cfg = compose(config_name="config")
        
        # Test specific values
        assert isinstance(cfg.training.batch_size, int)
        assert cfg.training.batch_size > 0
        assert isinstance(cfg.training.learning_rate, float)
        assert cfg.training.learning_rate > 0
        assert isinstance(cfg.latent_dim, int)
        assert cfg.latent_dim > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

