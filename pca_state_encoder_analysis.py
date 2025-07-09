# PCA and Variance Analysis for State Encoder Features
# =====================================================
#
# Checklist:
# 1. Encode a batch of states using the state encoder
# 2. Arrange encoded features in a matrix (samples x features)
# 3. Apply PCA and other dimensionality reduction (e.g., t-SNE, UMAP)
# 4. Plot PCA (2D/3D) and other projections
# 5. Analyze explained variance (per component, cumulative)
# 6. Analyze total variance, variance per feature, and covariance between features
#
# Timeline:
# - Step 1: Load config, model, and data buffer
# - Step 2: Encode a batch of states, collect features
# - Step 3: Arrange features in a matrix and transpose as needed
# - Step 4: Apply PCA, t-SNE, UMAP (if available)
# - Step 5: Plot projections and explained variance
# - Step 6: Compute and analyze total variance, per-feature variance, and covariance matrix
# - Step 7: Summarize findings

# Implementation will proceed step by step below... 

import os
import sys
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.state_encoder import StateEncoder
from train_step_distance_encoder import StepDistanceDataset

# ---- Step 1: Load config, model, and data buffer ----
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_state_encoder_from_checkpoint(checkpoint_path, config):
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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['state_encoder'])
    else:
        raise KeyError("No 'state_encoder' key found in checkpoint")
    return encoder

# Example config/checkpoint/buffer paths (edit as needed)
CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = "best_model_next_state_predictor.pth"
BUFFER_PATH = "data/buffer_500.pt"

# Load config
def main():
    print("Loading config...")
    config = load_config(CONFIG_PATH)
    print("Loading state encoder from checkpoint...")
    encoder = load_state_encoder_from_checkpoint(CHECKPOINT_PATH, config)
    encoder.eval()
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    encoder.to(device)
    print(f"Using device: {device}")
    # Load buffer as dataset
    encoder_params = config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    dataset = StepDistanceDataset(
        buffer_path=BUFFER_PATH,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='full'
    )
    print(f"Loaded dataset with {len(dataset)} samples")

    # ---- Step 2: Encode a batch of states and collect features ----
    N = 5000  # Number of states to sample for analysis (adjust as needed)
    print(f"Randomly selecting {N} unique states from the dataset...")
    total_samples = len(dataset)
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.permutation(total_samples)[:N]
    features = []
    meta = []
    encoder.eval()
    with torch.no_grad():
        for idx in selected_indices:
            sample = dataset[idx]
            state = sample['state'].unsqueeze(0).to(device)
            shape_h = sample['shape_h'].unsqueeze(0).to(device)
            shape_w = sample['shape_w'].unsqueeze(0).to(device)
            num_colors_grid = sample['num_colors_grid'].unsqueeze(0).to(device)
            most_present_color = sample['most_present_color'].unsqueeze(0).to(device)
            least_present_color = sample['least_present_color'].unsqueeze(0).to(device)
            # Encode state
            latent = encoder(
                state.to(torch.long),
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            features.append(latent.cpu().numpy().flatten())
            meta.append({
                'shape_h': shape_h.item(),
                'shape_w': shape_w.item(),
                'num_colors_grid': num_colors_grid.item(),
                'most_present_color': most_present_color.item(),
                'least_present_color': least_present_color.item()
            })
            if (len(features)) % 200 == 0:
                print(f"Encoded {len(features)}/{N} states...")
    features = np.stack(features, axis=0)  # Shape: (N, latent_dim)
    print(f"Feature matrix shape: {features.shape}")

    # ---- Step 3: Apply PCA, t-SNE, and UMAP (if available) ----
    print("Applying PCA...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print("PCA explained variance ratio (2D):", pca.explained_variance_ratio_)

    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    if HAS_UMAP:
        print("Applying UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        features_umap = reducer.fit_transform(features)
    else:
        features_umap = None

    # ---- Step 4: Plot 2D projections ----
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(features_pca[:, 0], features_pca[:, 1], s=10, alpha=0.7)
    plt.title('PCA (2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.subplot(1, 3, 2)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], s=10, alpha=0.7)
    plt.title('t-SNE (2D)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.subplot(1, 3, 3)
    if features_umap is not None:
        plt.scatter(features_umap[:, 0], features_umap[:, 1], s=10, alpha=0.7)
        plt.title('UMAP (2D)')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
    else:
        plt.text(0.5, 0.5, 'UMAP not installed', ha='center', va='center', fontsize=14)
        plt.title('UMAP (2D)')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig('pca_tsne_umap_2d.png', dpi=200)
    plt.show()

    # ---- Step 5: Analyze explained variance, total variance, feature variance, and covariance ----
    print("\nExplained variance by PCA components:")
    pca_full = PCA(n_components=min(features.shape[0], features.shape[1]))
    pca_full.fit(features)
    explained_var = pca_full.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)
    print("Explained variance ratio (first 10):", explained_var[:10])
    print("Cumulative explained variance (first 10):", cum_explained_var[:10])
    print(f"Total variance explained by first 2 components: {cum_explained_var[1]:.4f}")
    print(f"Total variance explained by all components: {cum_explained_var[-1]:.4f}")

    # Plot explained variance
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(explained_var)+1), explained_var, marker='o', label='Individual')
    plt.plot(np.arange(1, len(cum_explained_var)+1), cum_explained_var, marker='s', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_explained_variance.png', dpi=200)
    plt.show()

    # Total variance
    total_var = np.var(features, axis=0).sum()
    print(f"Total variance (sum of feature variances): {total_var:.4f}")

    # Variance per feature
    var_per_feature = np.var(features, axis=0)
    print("Variance per feature (first 10):", var_per_feature[:10])
    print(f"Mean variance per feature: {var_per_feature.mean():.4f}")

    # Covariance matrix
    cov_matrix = np.cov(features, rowvar=False)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print("Covariance matrix (first 5x5 block):\n", cov_matrix[:5, :5])
    plt.figure(figsize=(6, 5))
    plt.imshow(cov_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Covariance')
    plt.title('Feature Covariance Matrix')
    plt.tight_layout()
    plt.savefig('feature_covariance_matrix.png', dpi=200)
    plt.show()

    print("\nAnalysis complete. See generated plots and printed statistics for details.")

if __name__ == "__main__":
    main() 