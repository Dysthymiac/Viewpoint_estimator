"""Utility functions for PCA operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA


def save_pca_model(pca: IncrementalPCA, filepath: Path) -> None:
    """Save fitted PCA model to disk."""
    if not hasattr(pca, 'components_'):
        raise ValueError("PCA not fitted. Nothing to save.")
    
    np.savez_compressed(
        filepath,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        mean=pca.mean_,
        n_components=pca.n_components_
    )


def load_pca_model(filepath: Path) -> IncrementalPCA:
    """Load fitted PCA model from disk."""
    data = np.load(filepath)
    
    # Create new PCA with loaded parameters
    n_components = int(data['n_components'])
    pca = IncrementalPCA(n_components=n_components)
    
    # Set fitted parameters
    pca.components_ = data['components']
    pca.explained_variance_ = data['explained_variance']
    pca.explained_variance_ratio_ = data['explained_variance_ratio']
    pca.mean_ = data['mean']
    pca.n_components_ = n_components
    pca.n_features_in_ = data['components'].shape[1]
    
    return pca