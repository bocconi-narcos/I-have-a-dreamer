#!/usr/bin/env python3
"""
State Distance Analysis Script

This script analyzes whether the encoder from the selection predictor model
captures information about the step distance between states. It loads the
encoder from best_model_selection_predictor.pth and evaluates it on data
from data/buffer_10.pt.

For each transition in the dataset, it:
1. Encodes the current state and target state
2. Calculates multiple similarity metrics between encoded vectors
3. Plots step distance (y-axis) vs similarity metrics (x-axis)

The goal is to assess if the encoder captures distance information as a
proxy for effective world model learning.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.state_encoder import StateEncoder
from train_step_distance_encoder import StepDistanceDataset

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_encoder_from_checkpoint(checkpoint_path, config):
    """Load the state encoder from the selection predictor checkpoint."""
    
    # Initialize encoder with config parameters
    encoder_params = config['encoder_params']
    latent_dim = config['latent_dim']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state encoder weights
    if 'state_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['state_encoder'])
        print("Successfully loaded state encoder weights")
    else:
        raise KeyError("No 'state_encoder' key found in checkpoint")
    
    return encoder

def compute_cosine_similarity(x, y):
    """Compute cosine similarity between two vectors."""
    # Normalize vectors
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    
    # Compute cosine similarity
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    return cos_sim

def compute_euclidean_distance(x, y):
    """Compute Euclidean distance between two vectors."""
    return torch.norm(x - y, p=2, dim=-1)

def compute_euclidean_similarity(x, y):
    """Convert Euclidean distance to similarity using exponential decay."""
    distance = compute_euclidean_distance(x, y)
    # Convert to similarity: similarity = exp(-distance)
    # This ensures similarity is in [0, 1] range and decreases with distance
    return torch.exp(-distance)

def compute_euclidean_similarity_inverse(x, y):
    """Convert Euclidean distance to similarity using inverse relationship."""
    distance = compute_euclidean_distance(x, y)
    # Convert to similarity: similarity = 1 / (1 + distance)
    return 1.0 / (1.0 + distance)

def compute_manhattan_distance(x, y):
    """Compute Manhattan (L1) distance between two vectors."""
    return torch.norm(x - y, p=1, dim=-1)

def compute_manhattan_similarity(x, y):
    """Convert Manhattan distance to similarity."""
    distance = compute_manhattan_distance(x, y)
    return 1.0 / (1.0 + distance)

def compute_all_similarities(x, y):
    """Compute all similarity metrics between two vectors."""
    results = {}
    
    # Cosine similarity
    results['cosine_similarity'] = compute_cosine_similarity(x, y)
    
    # Euclidean metrics
    results['euclidean_distance'] = compute_euclidean_distance(x, y)
    results['euclidean_similarity_exp'] = compute_euclidean_similarity(x, y)
    results['euclidean_similarity_inv'] = compute_euclidean_similarity_inverse(x, y)
    
    # Manhattan metrics
    results['manhattan_distance'] = compute_manhattan_distance(x, y)
    results['manhattan_similarity'] = compute_manhattan_similarity(x, y)
    
    return results

def analyze_state_distances_multi_metric(encoder, dataset, device, max_samples=None):
    """
    Analyze the relationship between step distances and multiple similarity metrics.
    
    Args:
        encoder: The state encoder model
        dataset: The StepDistanceDataset containing state pairs and distances
        device: PyTorch device
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        dict: Analysis results containing similarities, distances, and metrics
    """
    encoder.eval()
    
    # Store results for each metric
    metric_results = {
        'cosine_similarity': [],
        'euclidean_distance': [],
        'euclidean_similarity_exp': [],
        'euclidean_similarity_inv': [],
        'manhattan_distance': [],
        'manhattan_similarity': []
    }
    step_distances = []
    
    print(f"Analyzing {len(dataset)} samples with multiple metrics...")
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
            
            # Extract data
            state = sample['state'].unsqueeze(0).to(device)
            target_state = sample['target_state'].unsqueeze(0).to(device)
            step_distance = sample['step_distance_to_target'].item()
            
            # Extract metadata
            shape_h = sample['shape_h'].unsqueeze(0).to(device)
            shape_w = sample['shape_w'].unsqueeze(0).to(device)
            num_colors_grid = sample['num_colors_grid'].unsqueeze(0).to(device)
            most_present_color = sample['most_present_color'].unsqueeze(0).to(device)
            least_present_color = sample['least_present_color'].unsqueeze(0).to(device)
            
            # Encode states
            state_encoding = encoder(
                state,
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            target_encoding = encoder(
                target_state,
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            # Compute all similarities
            similarities = compute_all_similarities(state_encoding, target_encoding)
            
            # Store results
            for metric, value in similarities.items():
                metric_results[metric].append(value.cpu().item())
            
            step_distances.append(step_distance)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{min(len(dataset), max_samples or len(dataset))} samples")
    
    # Convert to numpy arrays and compute correlations
    results = {}
    step_distances = np.array(step_distances)
    
    for metric, values in metric_results.items():
        values = np.array(values)
        
        # Compute correlation metrics
        pearson_corr, pearson_p = pearsonr(values, step_distances)
        spearman_corr, spearman_p = spearmanr(values, step_distances)
        
        # Compute R-squared for linear relationship
        try:
            r_squared = r2_score(step_distances, values)
        except:
            r_squared = 0.0
        
        results[metric] = {
            'values': values,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'r_squared': r_squared,
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'n_samples': len(values)
        }
    
    results['step_distances'] = step_distances
    return results

def create_multi_metric_visualizations(results, save_dir='distance_analysis'):
    """Create comprehensive visualizations for multiple similarity metrics."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    step_distances = results['step_distances']
    
    # Define metrics to plot and their expected relationships
    metrics_info = {
        'cosine_similarity': {
            'title': 'Cosine Similarity',
            'expected': 'negative',
            'description': 'Angle between vectors'
        },
        'euclidean_distance': {
            'title': 'Euclidean Distance',
            'expected': 'positive',
            'description': 'L2 distance between vectors'
        },
        'euclidean_similarity_exp': {
            'title': 'Euclidean Similarity (Exp)',
            'expected': 'negative',
            'description': 'exp(-distance)'
        },
        'euclidean_similarity_inv': {
            'title': 'Euclidean Similarity (Inv)',
            'expected': 'negative',
            'description': '1/(1+distance)'
        },
        'manhattan_distance': {
            'title': 'Manhattan Distance',
            'expected': 'positive',
            'description': 'L1 distance between vectors'
        },
        'manhattan_similarity': {
            'title': 'Manhattan Similarity',
            'expected': 'negative',
            'description': '1/(1+L1_distance)'
        }
    }
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Metric State Distance Analysis', fontsize=16)
    
    axes = axes.flatten()
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        metric_values = results[metric]['values']
        correlation = results[metric]['pearson_correlation']
        p_value = results[metric]['pearson_p_value']
        
        # Scatter plot
        scatter = ax.scatter(metric_values, step_distances, 
                           c=step_distances, cmap='viridis', alpha=0.6, s=20)
        
        # Regression line
        z = np.polyfit(metric_values, step_distances, 1)
        p = np.poly1d(z)
        ax.plot(metric_values, p(metric_values), "r--", alpha=0.8, linewidth=2)
        
        # Formatting
        ax.set_xlabel(info['title'])
        ax.set_ylabel('Step Distance')
        ax.set_title(f"{info['title']}\nr={correlation:.3f}, p={p_value:.4f}")
        ax.grid(True, alpha=0.3)
        
        # Add interpretation
        expected = info['expected']
        actual_direction = 'positive' if correlation > 0 else 'negative'
        color = 'green' if expected == actual_direction else 'red'
        ax.text(0.02, 0.98, f"Expected: {expected}\nActual: {actual_direction}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multi_metric_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'multi_metric_analysis.pdf'), bbox_inches='tight')
    
    # Create density plots for key metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Density Analysis: Key Metrics', fontsize=16)
    
    key_metrics = ['cosine_similarity', 'euclidean_distance', 'euclidean_similarity_exp', 'euclidean_similarity_inv']
    
    for i, metric in enumerate(key_metrics):
        ax = axes[i // 2, i % 2]
        metric_values = results[metric]['values']
        
        # Hexbin plot
        hb = ax.hexbin(metric_values, step_distances, gridsize=30, cmap='YlOrRd')
        ax.set_xlabel(metrics_info[metric]['title'])
        ax.set_ylabel('Step Distance')
        ax.set_title(f"{metrics_info[metric]['title']} Density")
        plt.colorbar(hb, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'density_analysis_multi_metric.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'density_analysis_multi_metric.pdf'), bbox_inches='tight')
    
    # Create distributions comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution Comparison', fontsize=16)
    
    axes = axes.flatten()
    
    for i, (metric, info) in enumerate(metrics_info.items()):
        ax = axes[i]
        metric_values = results[metric]['values']
        
        ax.hist(metric_values, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.set_xlabel(info['title'])
        ax.set_ylabel('Density')
        ax.set_title(f"{info['title']} Distribution")
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = results[metric]['mean']
        std_val = results[metric]['std']
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distributions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'distributions_comparison.pdf'), bbox_inches='tight')
    
    print(f"Multi-metric visualizations saved to {save_dir}/")

def print_multi_metric_analysis_summary(results):
    """Print a comprehensive summary of the multi-metric analysis results."""
    
    print("\n" + "="*100)
    print("MULTI-METRIC STATE DISTANCE ANALYSIS SUMMARY")
    print("="*100)
    
    step_distances = results['step_distances']
    print(f"Dataset Information:")
    print(f"  • Total samples analyzed: {len(step_distances):,}")
    print(f"  • Step distance range: [{step_distances.min():.0f}, {step_distances.max():.0f}]")
    print(f"  • Mean step distance: {step_distances.mean():.2f} ± {step_distances.std():.2f}")
    
    print(f"\nMetric Analysis:")
    print(f"{'Metric':<25} {'Pearson r':<10} {'P-value':<10} {'Expected':<10} {'Status':<15}")
    print("-" * 80)
    
    # Define expected relationships
    expected_relationships = {
        'cosine_similarity': 'negative',
        'euclidean_distance': 'positive',
        'euclidean_similarity_exp': 'negative',
        'euclidean_similarity_inv': 'negative',
        'manhattan_distance': 'positive',
        'manhattan_similarity': 'negative'
    }
    
    for metric in expected_relationships.keys():
        if metric in results:
            r = results[metric]['pearson_correlation']
            p = results[metric]['pearson_p_value']
            expected = expected_relationships[metric]
            actual = 'positive' if r > 0 else 'negative'
            
            # Status assessment
            if abs(r) < 0.1:
                status = "Very Weak"
            elif abs(r) < 0.3:
                status = "Weak"
            elif expected == actual:
                status = "Correct Direction"
            else:
                status = "Wrong Direction"
            
            print(f"{metric:<25} {r:<10.4f} {p:<10.6f} {expected:<10} {status:<15}")
    
    print(f"\nWorld Model Learning Assessment:")
    
    # Assess overall performance
    good_metrics = 0
    total_metrics = 0
    
    for metric in expected_relationships.keys():
        if metric in results:
            r = results[metric]['pearson_correlation']
            p = results[metric]['pearson_p_value']
            expected = expected_relationships[metric]
            actual = 'positive' if r > 0 else 'negative'
            
            total_metrics += 1
            if abs(r) > 0.1 and expected == actual and p < 0.05:
                good_metrics += 1
    
    performance_ratio = good_metrics / total_metrics if total_metrics > 0 else 0
    
    if performance_ratio > 0.5:
        print("  ✓ POSITIVE: Multiple metrics show expected relationships!")
        print("    The encoder captures meaningful distance information across different metrics.")
    elif performance_ratio > 0.2:
        print("  ⚠ MIXED: Some metrics show expected relationships, others don't.")
        print("    The encoder may capture partial distance information.")
    else:
        print("  ✗ LIMITED: Most metrics show weak or incorrect relationships.")
        print("    The encoder shows limited evidence of capturing distance information.")
    
    print(f"\nDetailed Metric Insights:")
    
    # Cosine vs Euclidean comparison
    if 'cosine_similarity' in results and 'euclidean_distance' in results:
        cos_r = results['cosine_similarity']['pearson_correlation']
        euc_r = results['euclidean_distance']['pearson_correlation']
        
        print(f"  • Cosine vs Euclidean:")
        print(f"    - Cosine similarity correlation: {cos_r:.4f}")
        print(f"    - Euclidean distance correlation: {euc_r:.4f}")
        
        if abs(cos_r) > abs(euc_r):
            print("    - Cosine similarity shows stronger relationship (captures angle better than magnitude)")
        else:
            print("    - Euclidean distance shows stronger relationship (captures magnitude better than angle)")
    
    # Distance vs Similarity comparison
    if 'euclidean_distance' in results and 'euclidean_similarity_exp' in results:
        dist_r = results['euclidean_distance']['pearson_correlation']
        sim_r = results['euclidean_similarity_exp']['pearson_correlation']
        
        print(f"  • Distance vs Similarity transformation:")
        print(f"    - Raw distance correlation: {dist_r:.4f}")
        print(f"    - Similarity (exp) correlation: {sim_r:.4f}")
        print(f"    - Transformation {'improves' if abs(sim_r) > abs(dist_r) else 'worsens'} the relationship")
    
    print("\n" + "="*100)

def main():
    """Main function to run the multi-metric state distance analysis."""
    
    # Configuration
    config_path = "../config.yaml"
    checkpoint_path = "../best_model_next_state_predictor.pth"
    buffer_path = "../data/buffer_500.pt"
    max_samples = 10000  # Limit samples for faster analysis, set to None for all
    
    print("Starting Multi-Metric State Distance Analysis...")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Buffer: {buffer_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load encoder
    encoder = load_encoder_from_checkpoint(checkpoint_path, config)
    encoder.to(device)
    
    # Create dataset
    encoder_params = config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    dataset = StepDistanceDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='full'
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Run multi-metric analysis
    results = analyze_state_distances_multi_metric(encoder, dataset, device, max_samples)
    
    # Print summary
    print_multi_metric_analysis_summary(results)
    
    # Create visualizations
    create_multi_metric_visualizations(results)
    
    # Save results
    save_path = "distance_analysis/multi_metric_results.npz"
    np.savez(save_path, **results)
    print(f"\nResults saved to {save_path}")
    
    print("\nMulti-metric analysis complete! Check the distance_analysis/ directory for visualizations.")

if __name__ == "__main__":
    main() 