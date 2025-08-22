"""Patch uncertainty estimation functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import entropy

from .component_normalization import normalize_components
from .patch_overlay import calculate_patch_bounds, apply_bilinear_smoothing


def calculate_entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate entropy-based uncertainty for each patch from probability distributions.
    Higher entropy = more uncertain (uniform distribution across components).
    
    Args:
        probabilities: Shape (n_patches, n_components) normalized probability distributions
        
    Returns:
        Uncertainty values of shape (n_patches,)
    """
    return np.array([entropy(patch_probs) for patch_probs in probabilities])


def calculate_max_ratio_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate uncertainty as 1 - (max_prob / second_max_prob) from probability distributions.
    Lower ratio = more uncertain (competing components).
    
    Args:
        probabilities: Shape (n_patches, n_components) normalized probability distributions
        
    Returns:
        Uncertainty values of shape (n_patches,)
    """
    # Sort each patch's probabilities in descending order
    sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
    
    # Calculate max to second-max ratio
    max_probs = sorted_probs[:, 0]
    second_max_probs = sorted_probs[:, 1] if sorted_probs.shape[1] > 1 else np.zeros_like(max_probs)
    
    # Avoid division by zero
    ratios = np.where(second_max_probs > 0, max_probs / second_max_probs, np.inf)
    
    # Convert to uncertainty (lower ratio = higher uncertainty)
    uncertainties = 1.0 / (1.0 + ratios)
    
    return uncertainties


def calculate_variance_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate uncertainty as variance of probability distributions.
    Lower variance = more uncertain (uniform distribution).
    
    Args:
        probabilities: Shape (n_patches, n_components) normalized probability distributions
        
    Returns:
        Uncertainty values of shape (n_patches,) (inverted: high value = high uncertainty)
    """
    # Calculate variance for each patch
    variances = np.var(probabilities, axis=1)
    
    # Invert variance to get uncertainty (low variance = high uncertainty)
    max_variance = np.max(variances)
    uncertainties = 1.0 - (variances / max_variance) if max_variance > 0 else np.ones_like(variances)
    
    return uncertainties


def calculate_uncertainty_from_probabilities(
    probabilities: np.ndarray,
    method: str = "entropy"
) -> np.ndarray:
    """
    Calculate patch uncertainty from normalized probability distributions.
    
    Args:
        probabilities: Shape (n_patches, n_components) normalized probability distributions
        method: "entropy", "max_ratio", or "variance"
        
    Returns:
        Uncertainty values of shape (n_patches,)
    """
    if method == "entropy":
        return calculate_entropy_from_probabilities(probabilities)
    elif method == "max_ratio":
        return calculate_max_ratio_from_probabilities(probabilities)
    elif method == "variance":
        return calculate_variance_from_probabilities(probabilities)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}. Use 'entropy', 'max_ratio', or 'variance'.")


def calculate_patch_uncertainty(
    components: np.ndarray,
    method: str = "entropy",
    component_indices: Optional[list[int]] = None,
    normalization_mode: str = "linear",
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convenient wrapper: process components and calculate uncertainty.
    
    Args:
        components: Shape (n_patches, n_components)
        method: "entropy", "max_ratio", or "variance"
        component_indices: Which components to consider
        normalization_mode: "linear" or "softmax"
        temperature: Temperature for softmax
        
    Returns:
        Uncertainty values of shape (n_patches,)
    """
    # Select components
    if component_indices is not None:
        selected_components = components[:, component_indices]
    else:
        selected_components = components
    
    # Normalize to get probability distributions
    probabilities = normalize_components(selected_components, normalization_mode, temperature)
    
    # Calculate uncertainty from probabilities
    return calculate_uncertainty_from_probabilities(probabilities, method)


def create_uncertainty_mask(
    image_size: tuple[int, int],
    patch_coordinates: np.ndarray,
    uncertainties: np.ndarray,
    relative_patch_size: float,
    smooth: bool = False
) -> np.ndarray:
    """
    Create uncertainty mask as grayscale array with optional smooth interpolation.
    
    Args:
        image_size: (width, height) of target image
        patch_coordinates: Shape (n_patches, 2) with relative (row, col) coordinates [0,1]
        uncertainties: Shape (n_patches,) uncertainty values
        relative_patch_size: Relative size of each patch [0,1]
        smooth: If True, apply bilinear interpolation for smooth heatmap
        
    Returns:
        Grayscale array of shape (height, width) with uncertainty values
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Calculate patch bounds using shared function
    y_starts, x_starts, patch_height, patch_width = calculate_patch_bounds(
        patch_coordinates, image_size, relative_patch_size
    )
    
    # Fill patch areas with uncertainty values
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        y_end = y_start + patch_height
        x_end = x_start + patch_width
        mask[y_start:y_end, x_start:x_end] = uncertainties[i]
    
    # Apply smoothing if requested
    mask = apply_bilinear_smoothing(mask, patch_coordinates, relative_patch_size, smooth)
    
    return mask