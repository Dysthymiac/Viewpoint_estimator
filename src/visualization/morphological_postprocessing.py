"""Morphological postprocessing for patch components."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import grey_opening, grey_closing
from skimage.morphology import disk, rectangle, square


def calculate_component_scaling(components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate per-component min and range for scaling.
    
    Args:
        components: Shape (n_patches, n_components)
        
    Returns:
        Tuple of (original_mins, original_ranges) for each component
    """
    original_mins = components.min(axis=0)
    original_maxs = components.max(axis=0)
    original_ranges = original_maxs - original_mins
    
    return original_mins, original_ranges


def normalize_components_for_morphology(components: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize components to [0,1] per-component globally.
    
    Args:
        components: Shape (n_patches, n_components)
        
    Returns:
        Tuple of (normalized_components, original_mins, original_ranges)
    """
    original_mins, original_ranges = calculate_component_scaling(components)
    
    # Avoid division by zero for constant components
    safe_ranges = np.where(original_ranges == 0, 1, original_ranges)
    
    normalized = (components - original_mins) / safe_ranges
    
    return normalized, original_mins, original_ranges


def restore_component_scaling(
    normalized: np.ndarray, 
    original_mins: np.ndarray, 
    original_ranges: np.ndarray
) -> np.ndarray:
    """
    Restore original component scaling after morphological operations.
    
    Args:
        normalized: Normalized components after morphological operations
        original_mins: Original minimum values per component
        original_ranges: Original ranges per component
        
    Returns:
        Components restored to original scaling
    """
    return normalized * original_ranges + original_mins



def create_morphological_kernel(kernel_size: int, kernel_shape: str = "disk") -> np.ndarray:
    """
    Create morphological kernel of specified size and shape.
    
    Args:
        kernel_size: Size of the kernel
        kernel_shape: "disk", "rectangle", "square"
        
    Returns:
        Binary kernel array
    """
    if kernel_shape == "disk":
        return disk(kernel_size)
    elif kernel_shape == "rectangle":
        return rectangle(kernel_size, kernel_size)
    elif kernel_shape == "square":
        return square(kernel_size)
    else:
        raise ValueError(f"Unknown kernel shape: {kernel_shape}. Use 'disk', 'rectangle', or 'square'.")


def apply_morphology_to_component(
    component_grid: np.ndarray, 
    operation: str, 
    kernel_size: int,
    kernel_shape: str = "disk"
) -> np.ndarray:
    """
    Apply single morphological operation to one component grid.
    
    Args:
        component_grid: 2D array of component values
        operation: "opening" or "closing"
        kernel_size: Size of morphological kernel
        kernel_shape: Shape of kernel ("disk", "rectangle", "square")
        
    Returns:
        Component grid after morphological operation
    """
    # Create kernel
    kernel = create_morphological_kernel(kernel_size, kernel_shape)
    
    # Apply grayscale morphological operation
    if operation == "opening":
        return grey_opening(component_grid, structure=kernel)
    elif operation == "closing":
        return grey_closing(component_grid, structure=kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'opening' or 'closing'.")


def apply_uncertainty_weighting(
    original_components: np.ndarray,
    cleaned_components: np.ndarray,
    uncertainty_weights: np.ndarray
) -> np.ndarray:
    """
    Blend original and cleaned components based on uncertainty weights.
    
    Args:
        original_components: Original component values
        cleaned_components: Morphologically cleaned components
        uncertainty_weights: Shape (n_patches,) - higher values = more uncertainty
        
    Returns:
        Weighted blend of original and cleaned components
    """
    # Convert uncertainty to cleanup strength (high uncertainty = more cleanup)
    uncertainty_weights = uncertainty_weights.reshape(-1, 1)  # Broadcast across components
    
    # Blend: low uncertainty preserves original, high uncertainty uses cleaned
    return (1 - uncertainty_weights) * original_components + uncertainty_weights * cleaned_components


