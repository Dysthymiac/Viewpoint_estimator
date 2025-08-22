"""Component normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_linear(components: np.ndarray) -> np.ndarray:
    """
    Linear normalization: components sum to 1.
    
    Args:
        components: Shape (n_patches, n_components)
        
    Returns:
        Normalized components where each row sums to 1
    """
    # Take absolute values to handle negative components
    abs_components = np.abs(components)
    
    # Normalize so each row sums to 1
    row_sums = abs_components.sum(axis=1, keepdims=True)
    
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    
    return abs_components / row_sums


def normalize_softmax(components: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax normalization with temperature scaling.
    
    Args:
        components: Shape (n_patches, n_components)
        temperature: Temperature parameter for softmax. Lower values = sharper distribution
        
    Returns:
        Softmax-normalized components where each row sums to 1
    """
    # Apply temperature scaling  
    scaled_components = components / temperature
    
    # Subtract max for numerical stability
    max_vals = np.max(scaled_components, axis=1, keepdims=True)
    stable_components = scaled_components - max_vals
    
    # Compute softmax
    exp_components = np.exp(stable_components)
    exp_sums = np.sum(exp_components, axis=1, keepdims=True)
    
    return exp_components / exp_sums


def normalize_components(
    components: np.ndarray, 
    mode: str = "linear", 
    temperature: float = 1.0
) -> np.ndarray:
    """
    Normalize component contributions for visualization.
    
    Args:
        components: Shape (n_patches, n_components)
        mode: "linear" or "softmax"
        temperature: Temperature for softmax (ignored for linear)
        
    Returns:
        Normalized components
    """
    if mode == "linear":
        return normalize_linear(components)
    elif mode == "softmax":
        return normalize_softmax(components, temperature)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Use 'linear' or 'softmax'.")