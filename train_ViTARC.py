#!/usr/bin/env python3
"""
ViTARC Comparative Training Script
=================================

This script runs comparative training experiments to evaluate different positional 
embedding combinations (APE, RPE, OPE) for the enhanced state encoder on the color 
predictor task. It systematically tests various configurations and tracks their 
performance to identify the optimal setup.

Author: AI Assistant
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import pandas as pd
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Import project modules
from src.models.state_encoder import EnhancedStateEncoder
from src.models.predictors.color_predictor import ColorPredictor
from src.models.action_embed import ActionEmbedder
from src.data import ReplayBufferDataset

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class PositionalEmbeddingConfig:
    """Configuration for positional embedding experiments"""
    name: str
    ape_type: str
    rpe_type: str
    rpe_abs: bool
    use_OPE: bool
    ape_mixer_strategy: str
    description: str

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    patience: int = 25
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    monitor_metric: str = "val_loss"  # "val_loss" or "val_acc"
    mode: str = "min"  # "min" for loss, "max" for accuracy
    min_epochs: int = 10  # Minimum epochs before early stopping can trigger
    max_epochs_without_improvement: int = 50  # Hard stop after this many epochs
    use_adaptive_patience: bool = True  # Increase patience if improvement is slow but steady

@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config_name: str
    best_val_loss: float
    best_val_acc: float
    final_train_loss: float
    training_time: float
    num_epochs: int
    num_parameters: int
    config_details: Dict
    best_epoch: int
    early_stopped: bool
    stopping_reason: str
    training_curve: Dict[str, List[float]]

# ============================================================================
# Experiment Configurations
# ============================================================================

def get_experiment_configurations() -> List[PositionalEmbeddingConfig]:
    """Define comprehensive set of positional embedding configurations to test"""
    configs = [
        # Baseline configurations
        PositionalEmbeddingConfig(
            name="baseline_no_pe",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="default",
            description="Basic 2D sinusoidal APE (baseline)"
        ),
        PositionalEmbeddingConfig(
            name="baseline_learned_ape",
            ape_type="LearnedAPE",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="default",
            description="Standard learned absolute positional embeddings"
        ),
        PositionalEmbeddingConfig(
            name="baseline_sinusoidal_ape",
            ape_type="SinusoidalAPE",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="default",
            description="Standard sinusoidal absolute positional embeddings"
        ),
        
        # 2D APE configurations
        PositionalEmbeddingConfig(
            name="ape_2d_basic",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="default",
            description="2D sinusoidal APE with default mixing"
        ),
        PositionalEmbeddingConfig(
            name="ape_2d_weighted_sum",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="2D sinusoidal APE with weighted sum mixing"
        ),
        PositionalEmbeddingConfig(
            name="ape_2d_learnable_scaling",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="learnable_scaling",
            description="2D sinusoidal APE with learnable scaling"
        ),
        PositionalEmbeddingConfig(
            name="ape_2d_layer_norm",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="layer_norm",
            description="2D sinusoidal APE with layer norm mixing"
        ),
        
        # RPE configurations
        PositionalEmbeddingConfig(
            name="rpe_two_slope_alibi",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=True,
            use_OPE=False,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Two-slope Alibi RPE with 2D APE"
        ),
        PositionalEmbeddingConfig(
            name="rpe_four_diag_slope_alibi",
            ape_type="SinusoidalAPE2D",
            rpe_type="Four-diag-slope-Alibi",
            rpe_abs=True,
            use_OPE=False,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Four-diagonal-slope Alibi RPE with 2D APE"
        ),
        PositionalEmbeddingConfig(
            name="rpe_two_slope_no_abs",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=False,
            use_OPE=False,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Two-slope Alibi RPE without absolute distance"
        ),
        
        # OPE configurations
        PositionalEmbeddingConfig(
            name="ope_basic",
            ape_type="SinusoidalAPE2D",
            rpe_type="None",
            rpe_abs=False,
            use_OPE=True,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Object positional embeddings with 2D APE"
        ),
        PositionalEmbeddingConfig(
            name="ope_with_rpe",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=True,
            use_OPE=True,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Object positional embeddings with RPE and 2D APE"
        ),
        PositionalEmbeddingConfig(
            name="ope_with_four_diag_rpe",
            ape_type="SinusoidalAPE2D",
            rpe_type="Four-diag-slope-Alibi",
            rpe_abs=True,
            use_OPE=True,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Object positional embeddings with Four-diag RPE"
        ),
        
        # Advanced mixer strategies
        PositionalEmbeddingConfig(
            name="positional_attention",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=True,
            use_OPE=False,
            ape_mixer_strategy="positional_attention",
            description="Positional attention mixing with RPE"
        ),
        PositionalEmbeddingConfig(
            name="hardcoded_normalization",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=True,
            use_OPE=False,
            ape_mixer_strategy="hardcoded_normalization",
            description="Hardcoded normalization mixing with RPE"
        ),
        
        # Best combination candidates
        PositionalEmbeddingConfig(
            name="full_vitarc",
            ape_type="SinusoidalAPE2D",
            rpe_type="Two-slope-Alibi",
            rpe_abs=True,
            use_OPE=True,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Full ViTARC: 2D APE + RPE + OPE"
        ),
        PositionalEmbeddingConfig(
            name="full_vitarc_four_diag",
            ape_type="SinusoidalAPE2D",
            rpe_type="Four-diag-slope-Alibi",
            rpe_abs=True,
            use_OPE=True,
            ape_mixer_strategy="weighted_sum_no_norm",
            description="Full ViTARC with Four-diagonal RPE"
        ),
    ]
    
    return configs

# ============================================================================
# Early Stopping Implementation
# ============================================================================

class EarlyStopping:
    """Enhanced early stopping with multiple strategies"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best_score = None
        self.epochs_no_improve = 0
        self.total_epochs_no_improve = 0
        self.best_epoch = 0
        self.should_stop = False
        self.stopping_reason = ""
        self.history = []
        self.best_weights = None
        
        # For adaptive patience
        self.recent_improvements = []
        self.adaptive_patience_factor = 1.0
        
    def __call__(self, epoch: int, current_score: float, model_state: Dict = None) -> bool:
        """
        Check if training should stop early
        
        Args:
            epoch: Current epoch number
            current_score: Current validation score (loss or accuracy)
            model_state: Current model state dict for saving best weights
            
        Returns:
            bool: True if training should stop
        """
        self.history.append(current_score)
        
        # Determine if current score is better
        if self.config.mode == "min":
            is_better = (self.best_score is None or 
                        current_score < self.best_score - self.config.min_delta)
        else:  # mode == "max"
            is_better = (self.best_score is None or 
                        current_score > self.best_score + self.config.min_delta)
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.total_epochs_no_improve = 0
            
            # Save best weights if provided
            if model_state is not None and self.config.restore_best_weights:
                self.best_weights = {}
                for model_name, state_dict in model_state.items():
                    self.best_weights[model_name] = {k: v.clone().cpu() for k, v in state_dict.items()}
            
            # Track improvement for adaptive patience
            if len(self.history) > 1:
                improvement = abs(current_score - self.history[-2])
                self.recent_improvements.append(improvement)
                if len(self.recent_improvements) > 10:
                    self.recent_improvements.pop(0)
                    
        else:
            self.epochs_no_improve += 1
            self.total_epochs_no_improve += 1
        
        # Check stopping conditions
        return self._check_stopping_conditions(epoch)
    
    def _check_stopping_conditions(self, epoch: int) -> bool:
        """Check various stopping conditions"""
        
        # Don't stop before minimum epochs
        if epoch < self.config.min_epochs:
            return False
        
        # Hard stop after max epochs without improvement
        if self.total_epochs_no_improve >= self.config.max_epochs_without_improvement:
            self.should_stop = True
            self.stopping_reason = f"No improvement for {self.config.max_epochs_without_improvement} epochs (hard stop)"
            return True
        
        # Calculate effective patience (with adaptation if enabled)
        effective_patience = self.config.patience
        if self.config.use_adaptive_patience and len(self.recent_improvements) > 5:
            # If improvements are small but consistent, increase patience
            avg_improvement = sum(self.recent_improvements) / len(self.recent_improvements)
            if avg_improvement > 0 and avg_improvement < self.config.min_delta * 2:
                effective_patience = int(self.config.patience * 1.5)
        
        # Standard patience-based stopping
        if self.epochs_no_improve >= effective_patience:
            self.should_stop = True
            self.stopping_reason = f"No improvement for {effective_patience} epochs (patience={self.config.patience})"
            return True
        
        return False
    
    def get_best_weights(self) -> Dict:
        """Get the best model weights"""
        return self.best_weights if self.best_weights is not None else {}
    
    def get_summary(self) -> Dict:
        """Get summary of early stopping behavior"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_epochs_no_improve': self.total_epochs_no_improve,
            'stopped_early': self.should_stop,
            'stopping_reason': self.stopping_reason,
            'effective_patience': self.config.patience * self.adaptive_patience_factor
        }

# ============================================================================
# Utility Functions
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert indices to one-hot encoding"""
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_results(results: List[ExperimentResult], output_dir: str):
    """Save experiment results to JSON and CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    results_dict = []
    for result in results:
        results_dict.append({
            'config_name': result.config_name,
            'best_val_loss': result.best_val_loss,
            'best_val_acc': result.best_val_acc,
            'final_train_loss': result.final_train_loss,
            'training_time': result.training_time,
            'num_epochs': result.num_epochs,
            'num_parameters': result.num_parameters,
            'config_details': result.config_details
        })
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame([
        {
            'config_name': r.config_name,
            'best_val_loss': r.best_val_loss,
            'best_val_acc': r.best_val_acc,
            'final_train_loss': r.final_train_loss,
            'training_time': r.training_time,
            'num_epochs': r.num_epochs,
            'num_parameters': r.num_parameters,
            'best_epoch': r.best_epoch,
            'early_stopped': r.early_stopped,
            'stopping_reason': r.stopping_reason,
            'convergence_efficiency': r.best_epoch / r.num_epochs,  # How quickly it converged
            'description': r.config_details.get('description', ''),
            'ape_type': r.config_details.get('ape_type', ''),
            'rpe_type': r.config_details.get('rpe_type', ''),
            'use_OPE': r.config_details.get('use_OPE', False),
            'ape_mixer_strategy': r.config_details.get('ape_mixer_strategy', '')
        }
        for r in results
    ])
    
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    print(f"Results saved to {output_dir}")

# ============================================================================
# Training Functions
# ============================================================================

def evaluate_model(
    model: nn.Module,
    encoder: nn.Module,
    action_embedder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_color_selection_fns: int
) -> Tuple[float, float]:
    """Evaluate model on validation set"""
    model.eval()
    encoder.eval()
    action_embedder.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc='Validating', leave=False, ncols=80)
        
        for batch in val_pbar:
            state = batch['state'].to(device)
            action_colour = batch['action_colour'].to(device)
            target_colour = batch['colour'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            
            # Get latent representation
            latent = encoder(
                state,
                shape_h=shape_h,
                shape_w=shape_w,
                most_common_color=most_present_color,
                least_common_color=least_present_color,
                num_unique_colors=num_colors_grid
            )
            
            # Get action embedding
            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            action_embedding = action_embedder(action_colour_onehot)
            
            # Get prediction
            logits = model(latent, action_embedding)
            loss = criterion(logits, target_colour)
            
            total_loss += loss.item() * state.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_colour).sum().item()
            total += state.size(0)
            
            current_acc = correct / total if total > 0 else 0
            val_pbar.set_postfix({'Acc': f'{current_acc:.3f}'})
        
        val_pbar.close()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_single_configuration(
    config: PositionalEmbeddingConfig,
    base_config: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    experiment_dir: str,
    use_wandb: bool = True
) -> ExperimentResult:
    """Train a single configuration and return results"""
    print(f"\n{'='*80}")
    print(f"Training Configuration: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Initialize wandb for this experiment
    if use_wandb:
        wandb.init(
            project="vitarc_comparison",
            name=config.name,
            config={
                **base_config,
                'pe_config': config.__dict__
            },
            reinit=True
        )
    
    # Extract configuration parameters
    latent_dim = base_config['latent_dim']
    encoder_params = base_config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    # Create enhanced state encoder with specific PE configuration
    state_encoder = EnhancedStateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params,
        ape_type=config.ape_type,
        rpe_type=config.rpe_type,
        rpe_abs=config.rpe_abs,
        use_OPE=config.use_OPE,
        ape_mixer_strategy=config.ape_mixer_strategy
    ).to(device)
    
    # Create action embedder and color predictor
    action_embedder = ActionEmbedder(
        num_actions=base_config['action_embedders']['action_color_embedder']['num_actions'],
        embed_dim=base_config['action_embedders']['action_color_embedder']['embed_dim'],
        dropout_p=0.1
    ).to(device)
    
    color_predictor = ColorPredictor(
        latent_dim=latent_dim,
        num_colors=11,
        hidden_dim=base_config['color_predictor']['hidden_dim'],
        action_embedding_dim=base_config['action_embedders']['action_color_embedder']['embed_dim']
    ).to(device)
    
    # Count parameters
    total_params = count_parameters(state_encoder) + count_parameters(action_embedder) + count_parameters(color_predictor)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(state_encoder.parameters()) + 
        list(action_embedder.parameters()) + 
        list(color_predictor.parameters()),
        lr=base_config['learning_rate']
    )
    
    # Enhanced early stopping setup - configurable via config file
    early_stop_params = base_config.get('early_stopping', {})
    early_stopping_config = EarlyStoppingConfig(
        patience=early_stop_params.get('patience', 15),
        min_delta=early_stop_params.get('min_delta', 1e-5),
        restore_best_weights=early_stop_params.get('restore_best_weights', True),
        monitor_metric=early_stop_params.get('monitor_metric', 'val_loss'),
        mode=early_stop_params.get('mode', 'min'),
        min_epochs=early_stop_params.get('min_epochs', 3),
        max_epochs_without_improvement=early_stop_params.get('max_epochs_without_improvement', 40),
        use_adaptive_patience=early_stop_params.get('use_adaptive_patience', False)
    )
    early_stopping = EarlyStopping(early_stopping_config)
    
    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    final_train_loss = 0.0
    best_epoch = 0
    num_epochs = min(base_config['num_epochs'], 150)  # Increased cap for better convergence
    
    # Training curves for analysis
    training_curves = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    print(f"Starting training for up to {num_epochs} epochs...")
    print(f"Early stopping: patience={early_stopping_config.patience}, min_delta={early_stopping_config.min_delta}")
    
    epoch_pbar = tqdm(range(num_epochs), desc=f'Training {config.name}', ncols=120)
    
    for epoch in epoch_pbar:
        # Training phase
        state_encoder.train()
        action_embedder.train()
        color_predictor.train()
        
        total_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                         leave=False, ncols=100)
        
        for batch in train_pbar:
            state = batch['state'].to(device)
            action_colour = batch['action_colour'].to(device)
            target_colour = batch['colour'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            
            # Forward pass
            latent = state_encoder(
                state,
                shape_h=shape_h,
                shape_w=shape_w,
                most_common_color=most_present_color,
                least_common_color=least_present_color,
                num_unique_colors=num_colors_grid
            )
            
            action_colour_onehot = one_hot(action_colour, base_config['action_embedders']['action_color_embedder']['num_actions'])
            action_embedding = action_embedder(action_colour_onehot)
            logits = color_predictor(latent, action_embedding)
            
            loss = criterion(logits, target_colour)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + 
                list(action_embedder.parameters()) + 
                list(color_predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_pbar.close()
        
        # Validation phase
        avg_train_loss = total_loss / num_batches
        val_loss, val_acc = evaluate_model(
            color_predictor, state_encoder, action_embedder, val_loader, device, criterion,
            base_config['action_embedders']['action_color_embedder']['num_actions']
        )
        
        # Record training curves
        training_curves['train_loss'].append(avg_train_loss)
        training_curves['val_loss'].append(val_loss)
        training_curves['val_acc'].append(val_acc)
        training_curves['epochs'].append(epoch + 1)
        
        # Update best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Create model state for early stopping
        current_model_state = {
            'state_encoder': state_encoder.state_dict(),
            'action_embedder': action_embedder.state_dict(),
            'color_predictor': color_predictor.state_dict()
        }
        
        # Check early stopping
        should_stop = early_stopping(epoch + 1, val_loss, current_model_state)
        early_stop_summary = early_stopping.get_summary()
        
        # Update progress
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Best': f'{best_val_loss:.4f}',
            'No Improve': f'{early_stopping.epochs_no_improve}'
        })
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'epochs_no_improve': early_stopping.epochs_no_improve,
                'early_stop_score': early_stopping.best_score
            })
        
        final_train_loss = avg_train_loss
        
        # Early stopping check
        if should_stop:
            tqdm.write(f"Early stopping at epoch {epoch+1}: {early_stopping.stopping_reason}")
            break
    
    epoch_pbar.close()
    
    # Restore best weights if early stopping was used
    best_weights = early_stopping.get_best_weights()
    if best_weights and early_stopping_config.restore_best_weights:
        print(f"Restoring best weights from epoch {early_stopping.best_epoch}")
        
        # Load best weights back to models
        if 'state_encoder' in best_weights:
            state_encoder.load_state_dict({k: v.to(device) for k, v in best_weights['state_encoder'].items()})
        if 'action_embedder' in best_weights:
            action_embedder.load_state_dict({k: v.to(device) for k, v in best_weights['action_embedder'].items()})
        if 'color_predictor' in best_weights:
            color_predictor.load_state_dict({k: v.to(device) for k, v in best_weights['color_predictor'].items()})
    
    # Final evaluation with best weights
    final_val_loss, final_val_acc = evaluate_model(
        color_predictor, state_encoder, action_embedder, val_loader, device, criterion,
        base_config['action_embedders']['action_color_embedder']['num_actions']
    )
    
    # Update best metrics if final evaluation is better
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
    if final_val_acc > best_val_acc:
        best_val_acc = final_val_acc
    
    # Save the final best model
    model_path = os.path.join(experiment_dir, f'best_model_{config.name}.pth')
    torch.save({
        'state_encoder': state_encoder.state_dict(),
        'action_embedder': action_embedder.state_dict(),
        'color_predictor': color_predictor.state_dict(),
        'config': config.__dict__,
        'val_loss': best_val_loss,
        'val_acc': best_val_acc,
        'best_epoch': early_stopping.best_epoch,
        'training_curves': training_curves,
        'early_stopping_summary': early_stopping.get_summary()
    }, model_path)
    
    training_time = time.time() - start_time
    early_stop_summary = early_stopping.get_summary()
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {early_stopping.best_epoch})")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Early stopping: {early_stop_summary['stopping_reason']}")
    
    if use_wandb:
        wandb.log({
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'total_parameters': total_params,
            'best_epoch': early_stopping.best_epoch,
            'total_epochs': epoch + 1,
            'early_stopped': early_stopping.should_stop
        })
        wandb.finish()
    
    return ExperimentResult(
        config_name=config.name,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        final_train_loss=final_train_loss,
        training_time=training_time,
        num_epochs=epoch+1,
        num_parameters=total_params,
        config_details=config.__dict__,
        best_epoch=early_stopping.best_epoch,
        early_stopped=early_stopping.should_stop,
        stopping_reason=early_stop_summary['stopping_reason'],
        training_curve=training_curves
    )

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function that runs all experiments"""
    print("="*80)
    print("ViTARC Comparative Training Script")
    print("="*80)
    
    # Load base configuration
    base_config = load_config()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "vitarc_experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup dataset
    buffer_path = base_config['buffer_path']
    encoder_params = base_config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    dataset = ReplayBufferDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=base_config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=base_config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=base_config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=base_config['num_arc_colors']-1,
        state_shape=state_shape,
        mode='color_only'
    )
    
    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=base_config['batch_size'], 
        shuffle=True, 
        num_workers=base_config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=base_config['batch_size'], 
        shuffle=False, 
        num_workers=base_config['num_workers']
    )
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Get experiment configurations
    configs = get_experiment_configurations()
    print(f"Running {len(configs)} experiments...")
    
    # Run experiments
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running experiment: {config.name}")
        
        try:
            result = train_single_configuration(
                config=config,
                base_config=base_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                experiment_dir=output_dir,
                use_wandb=True
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error in experiment {config.name}: {e}")
            continue
    
    # Save and analyze results
    save_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # Sort by validation accuracy
    results_sorted = sorted(results, key=lambda x: x.best_val_acc, reverse=True)
    
    print(f"{'Rank':<4} {'Configuration':<25} {'Val Acc':<10} {'Val Loss':<10} {'Best Ep':<8} {'Early Stop':<10} {'Time (s)':<10}")
    print("-" * 95)
    
    for i, result in enumerate(results_sorted):
        early_stop_indicator = "✓" if result.early_stopped else "✗"
        print(f"{i+1:<4} {result.config_name:<25} {result.best_val_acc:<10.4f} "
              f"{result.best_val_loss:<10.4f} {result.best_epoch:<8} {early_stop_indicator:<10} {result.training_time:<10.1f}")
    
    print("\nTop 3 configurations:")
    for i, result in enumerate(results_sorted[:3]):
        print(f"{i+1}. {result.config_name}: {result.best_val_acc:.4f} accuracy (epoch {result.best_epoch})")
        print(f"   Description: {result.config_details.get('description', 'N/A')}")
        print(f"   Early stopped: {result.early_stopped} ({result.stopping_reason})")
        convergence_eff = result.best_epoch / result.num_epochs
        print(f"   Convergence efficiency: {convergence_eff:.2f} (reached best at {convergence_eff*100:.1f}% of training)")
    
    # Early stopping analysis
    early_stopped_count = sum(1 for r in results if r.early_stopped)
    avg_best_epoch = sum(r.best_epoch for r in results) / len(results)
    avg_convergence_eff = sum(r.best_epoch / r.num_epochs for r in results) / len(results)
    
    print(f"\nEarly Stopping Analysis:")
    print(f"- Experiments early stopped: {early_stopped_count}/{len(results)} ({early_stopped_count/len(results)*100:.1f}%)")
    print(f"- Average best epoch: {avg_best_epoch:.1f}")
    print(f"- Average convergence efficiency: {avg_convergence_eff:.2f}")
    print(f"- Most common stopping reason: {max(set(r.stopping_reason for r in results), key=lambda x: sum(1 for r in results if r.stopping_reason == x))}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
