"""Patch overlay visualization functions."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from .component_colors import get_color_palette
from .component_normalization import normalize_components


def components_to_colors(
    components: np.ndarray,
    component_indices: Optional[list[int]] = None,
    normalization_mode: str = "linear",
    temperature: float = 1.0
) -> np.ndarray:
    """
    Convert PCA components to RGB colors.
    
    Args:
        components: Shape (n_patches, n_components)
        component_indices: Which components to use (default: use all or first few)
        normalization_mode: "linear" or "softmax"
        temperature: Temperature for softmax
        
    Returns:
        RGB colors array of shape (n_patches, 3)
    """
    # Select components to use
    if component_indices is not None:
        selected_components = components[:, component_indices]
        n_selected = len(component_indices)
    else:
        # Auto-select based on total number of components
        n_total = components.shape[1]
        if n_total <= 3:
            selected_components = components
            n_selected = n_total
        else:
            # Use first 3 components by default for >3 case
            selected_components = components[:, :3]
            n_selected = 3
    
    # Normalize components
    normalized = normalize_components(selected_components, normalization_mode, temperature)
    
    # Get color palette
    color_palette = get_color_palette(n_selected)
    
    # Convert normalized components to colors
    # Each patch color = weighted sum of palette colors
    colors = np.dot(normalized, color_palette)
    
    return colors


def calculate_patch_bounds(
    patch_coordinates: np.ndarray,
    image_size: tuple[int, int],
    relative_patch_size: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Calculate pixel bounds for patches with no gaps.
    
    Args:
        patch_coordinates: Shape (n_patches, 2) with relative (row, col) coordinates [0,1]
        image_size: (width, height) of target image
        relative_patch_size: Relative size of each patch [0,1]
        
    Returns:
        Tuple of (y_starts, x_starts, patch_height, patch_width)
    """
    width, height = image_size
    
    # Calculate patch size in pixels with ceiling to prevent gaps
    patch_height = math.ceil(relative_patch_size * height)
    patch_width = math.ceil(relative_patch_size * width)
    
    # Convert relative coordinates to pixel coordinates using rounding
    y_starts = np.round(patch_coordinates[:, 0] * height).astype(int)
    x_starts = np.round(patch_coordinates[:, 1] * width).astype(int)
    
    # Ensure bounds are within image
    y_starts = np.clip(y_starts, 0, height - patch_height)
    x_starts = np.clip(x_starts, 0, width - patch_width)
    
    return y_starts, x_starts, patch_height, patch_width


def apply_bilinear_smoothing(
    discrete_mask: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    smooth: bool = False
) -> np.ndarray:
    """
    Convert discrete patch mask to smooth bilinear interpolation.
    
    Args:
        discrete_mask: Discrete patch mask from create_patch_mask or create_uncertainty_mask
        patch_coordinates: Shape (n_patches, 2) with relative (row, col) coordinates [0,1]
        relative_patch_size: Relative size of each patch [0,1]
        smooth: If False, returns discrete_mask unchanged (pass-through)
        
    Returns:
        Smoothly interpolated mask with same shape as discrete_mask
    """
    if not smooth:
        return discrete_mask
    
    # Determine if this is RGB (3D) or grayscale (2D) mask
    is_rgb = len(discrete_mask.shape) == 3
    
    if is_rgb:
        height, width, channels = discrete_mask.shape
    else:
        height, width = discrete_mask.shape
        channels = 1
    
    # Calculate patch centers in pixel coordinates
    patch_centers_y = patch_coordinates[:, 0] * height
    patch_centers_x = patch_coordinates[:, 1] * width
    
    # Determine grid bounds based on patch positions
    min_y, max_y = patch_centers_y.min(), patch_centers_y.max()
    min_x, max_x = patch_centers_x.min(), patch_centers_x.max()
    
    # Estimate grid spacing from relative patch size
    grid_step_y = relative_patch_size * height
    grid_step_x = relative_patch_size * width
    
    # Create regular grid coordinates for interpolation
    grid_y = np.arange(min_y, max_y + grid_step_y, grid_step_y)
    grid_x = np.arange(min_x, max_x + grid_step_x, grid_step_x)
    
    # Extract patch values from discrete mask
    patch_values = []
    for i, (center_y, center_x) in enumerate(zip(patch_centers_y, patch_centers_x)):
        y_idx = int(round(center_y))
        x_idx = int(round(center_x))
        
        # Ensure indices are within bounds
        y_idx = max(0, min(y_idx, height - 1))
        x_idx = max(0, min(x_idx, width - 1))
        
        if is_rgb:
            patch_values.append(discrete_mask[y_idx, x_idx, :])
        else:
            patch_values.append(discrete_mask[y_idx, x_idx])
    
    patch_values = np.array(patch_values)
    
    # Map patch coordinates to grid indices
    grid_data = np.zeros((len(grid_y), len(grid_x), channels))
    
    for i, (center_y, center_x) in enumerate(zip(patch_centers_y, patch_centers_x)):
        # Find nearest grid indices
        grid_y_idx = np.argmin(np.abs(grid_y - center_y))
        grid_x_idx = np.argmin(np.abs(grid_x - center_x))
        
        if is_rgb:
            grid_data[grid_y_idx, grid_x_idx, :] = patch_values[i]
        else:
            grid_data[grid_y_idx, grid_x_idx, 0] = patch_values[i]
    
    # Create output coordinates for full image
    output_y = np.arange(height)
    output_x = np.arange(width)
    output_coords = np.meshgrid(output_y, output_x, indexing='ij')
    output_points = np.column_stack([output_coords[0].ravel(), output_coords[1].ravel()])
    
    # Initialize smooth mask
    if is_rgb:
        smooth_mask = np.zeros((height, width, channels))
    else:
        smooth_mask = np.zeros((height, width))
    
    # Interpolate each channel
    for c in range(channels):
        # Create interpolator for this channel
        interpolator = RegularGridInterpolator(
            (grid_y, grid_x), 
            grid_data[:, :, c],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Interpolate values at all output pixels
        interpolated_values = interpolator(output_points).reshape(height, width)
        
        if is_rgb:
            smooth_mask[:, :, c] = interpolated_values
        else:
            smooth_mask = interpolated_values
    
    return smooth_mask


def create_patch_mask(
    image_size: tuple[int, int],
    patch_coordinates: np.ndarray,
    patch_colors: np.ndarray,
    relative_patch_size: float,
    smooth: bool = False
) -> np.ndarray:
    """
    Create patch color mask as numpy array with optional smooth interpolation.
    
    Args:
        image_size: (width, height) of target image
        patch_coordinates: Shape (n_patches, 2) with relative (row, col) coordinates [0,1]
        patch_colors: Shape (n_patches, 3) RGB colors for each patch
        relative_patch_size: Relative size of each patch [0,1]
        smooth: If True, apply bilinear interpolation for smooth heatmap
        
    Returns:
        RGB array of shape (height, width, 3) with patch colors
    """
    width, height = image_size
    mask = np.zeros((height, width, 3), dtype=np.float32)
    
    # Calculate patch bounds using shared function
    y_starts, x_starts, patch_height, patch_width = calculate_patch_bounds(
        patch_coordinates, image_size, relative_patch_size
    )
    
    # Fill patch areas with colors
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        y_end = y_start + patch_height
        x_end = x_start + patch_width
        mask[y_start:y_end, x_start:x_end] = patch_colors[i]
    
    # Apply smoothing if requested
    mask = apply_bilinear_smoothing(mask, patch_coordinates, relative_patch_size, smooth)
    
    return mask


def blend_image_with_patches(
    image: Image.Image,
    patch_mask: np.ndarray,
    alpha: float = 0.6
) -> Image.Image:
    """
    Blend original image with patch color mask.
    
    Args:
        image: Original PIL Image
        patch_mask: RGB array of shape (height, width, 3) with patch colors
        alpha: Alpha blending factor (0=original image, 1=patches only)
        
    Returns:
        Blended PIL Image
    """
    # Convert image to numpy array
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Ensure image and mask have same dimensions
    if image_array.shape[:2] != patch_mask.shape[:2]:
        raise ValueError(f"Image shape {image_array.shape[:2]} doesn't match mask shape {patch_mask.shape[:2]}")
    
    # Blend image with patches
    blended = (1 - alpha) * image_array + alpha * patch_mask
    
    # Convert back to PIL Image
    blended_uint8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(blended_uint8)


def create_patch_overlay_image(
    image: Image.Image,
    components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    component_indices: Optional[list[int]] = None,
    normalization_mode: str = "linear",
    temperature: float = 1.0,
    alpha: float = 0.6,
    smooth: bool = False
) -> Image.Image:
    """
    Complete pipeline: create overlaid image with patch components.
    
    Args:
        image: Original PIL Image
        components: Shape (n_patches, n_components) PCA components
        patch_coordinates: Shape (n_patches, 2) with relative (row, col) coordinates [0,1]
        relative_patch_size: Relative size of each patch [0,1]
        component_indices: Which components to visualize
        normalization_mode: "linear" or "softmax"
        temperature: Temperature for softmax
        alpha: Alpha blending factor
        smooth: If True, apply bilinear interpolation for smooth heatmap
        
    Returns:
        PIL Image with patch overlay
    """
    # Convert components to colors
    patch_colors = components_to_colors(
        components, component_indices, normalization_mode, temperature
    )
    
    # Create patch color mask
    patch_mask = create_patch_mask(
        image.size, patch_coordinates, patch_colors, relative_patch_size, smooth
    )
    
    # Blend with original image
    return blend_image_with_patches(image, patch_mask, alpha)