"""Morphological utilities for animal detection and body part cleanup."""

from __future__ import annotations

from typing import List, Tuple, Optional
import math

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk, square

def apply_morphology(
    mask: np.ndarray,
    operations: List[str] = ["opening", "closing"],
    kernel_size: int = 1,
    kernel_shape: str = "disk"
) -> np.ndarray:
    """
    Apply morphological cleanup specifically for animal detection.
    
    Args:
        animal_mask: 2D binary mask of potential animal regions
        min_area_threshold: Minimum area as fraction of total image
        operations: List of morphological operations to apply
        kernel_size: Size of morphological kernel
        kernel_shape: Shape of kernel ("disk", "rectangle", "square")
        
    Returns:
        Cleaned binary mask
    """
    clean_mask = mask.copy().astype(bool)
    
    # Create kernel
    if kernel_shape == "disk":
        kernel = disk(kernel_size)
    elif kernel_shape == "square":
        kernel = square(kernel_size)
    else:
        kernel = disk(kernel_size)  # Default fallback
    
    # Apply morphological operations
    for operation in operations:
        if operation == "opening":
            clean_mask = binary_opening(clean_mask, kernel)
        elif operation == "closing":
            clean_mask = binary_closing(clean_mask, kernel)
            
    return clean_mask.astype(mask.dtype)

def create_mask_from_patches(
    patch_coordinates: np.ndarray, 
    patch_values: np.ndarray,
    relative_patch_size: float
) -> np.ndarray:
    """Create spatial mask using direct patch filling (working approach from detector)."""
    # Calculate image bounds
    grid_patch_coordinates = np.round(patch_coordinates/relative_patch_size)
    max_x = int(np.max(grid_patch_coordinates[:, 0]))+1
    max_y = int(np.max(grid_patch_coordinates[:, 1]))+1
    # Initialize mask
    mask = np.zeros((max_y, max_x), dtype=patch_values.dtype)
    for i, (x, y) in enumerate(grid_patch_coordinates):
        mask[int(y), int(x)] = patch_values[i]
    
    return mask

def apply_mask_to_patches(
    mask: np.ndarray,
    patch_coordinates: np.ndarray, 
    patch_values: np.ndarray,
    relative_patch_size: float
) -> np.ndarray:
    grid_patch_coordinates = np.round(patch_coordinates/relative_patch_size)
    new_values = patch_values.copy()
    for i, (x, y) in enumerate(grid_patch_coordinates):
        new_values[i] = mask[int(y), int(x)] * new_values[i]
    return new_values

def create_spatial_mask_from_patches(
    patch_coordinates: np.ndarray, 
    patch_values: np.ndarray,
    relative_patch_size: float
) -> np.ndarray:
    """Create spatial mask using direct patch filling (working approach from detector)."""
    # Calculate image bounds
    max_coord = np.max(patch_coordinates)
    min_coord = np.min(patch_coordinates)
    coord_range = max_coord - min_coord + relative_patch_size
    image_size = int(coord_range / relative_patch_size) + 1
    
    # Initialize mask
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Calculate patch dimensions
    patch_height = math.ceil(relative_patch_size * image_size)
    patch_width = math.ceil(relative_patch_size * image_size)
    
    # Convert coordinates to pixel positions
    y_starts = np.round((patch_coordinates[:, 0] - min_coord) / relative_patch_size).astype(int)
    x_starts = np.round((patch_coordinates[:, 1] - min_coord) / relative_patch_size).astype(int)
    
    # Ensure bounds
    y_starts = np.clip(y_starts, 0, image_size - patch_height)
    x_starts = np.clip(x_starts, 0, image_size - patch_width)
    
    # Fill patch areas
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        if patch_values[i] > 0:
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            mask[y_start:y_end, x_start:x_end] = patch_values[i]
    
    return (mask > 0.5).astype(np.uint8)


def create_continuous_regions(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    target_clusters: List[int],
    min_region_size: int = 1
) -> np.ndarray:
    """Create continuous binary regions using working direct patch filling."""
    # Combine target clusters
    combined_mask = np.zeros(len(patch_components))
    for cluster_id in target_clusters:
        if cluster_id < patch_components.shape[1]:
            combined_mask += patch_components[:, cluster_id]
    
    combined_mask = (combined_mask > 0.1).astype(float)
    
    # Create spatial mask using working approach
    binary_mask = create_spatial_mask_from_patches(patch_coordinates, combined_mask, relative_patch_size)
    
    # Remove small components
    cleaned_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_region_size)
    
    return cleaned_mask.astype(np.uint8)


def calculate_mask_properties(binary_mask: np.ndarray) -> dict:
    """
    Calculate basic properties of a binary mask.
    
    Args:
        binary_mask: 2D binary mask
        
    Returns:
        Dictionary with area, centroid, and bounding box
    """
    if not np.any(binary_mask):
        return {
            'area': 0,
            'centroid': (0, 0),
            'bbox': (0, 0, 0, 0),
            'area_ratio': 0.0
        }
    
    # Calculate area
    area = np.sum(binary_mask)
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    area_ratio = area / total_area
    
    # Calculate centroid
    rows, cols = np.where(binary_mask)
    centroid_row = np.mean(rows)
    centroid_col = np.mean(cols)
    
    # Calculate bounding box
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    return {
        'area': int(area),
        'centroid': (float(centroid_row), float(centroid_col)),
        'bbox': (int(min_row), int(min_col), int(max_row), int(max_col)),
        'area_ratio': float(area_ratio)
    }


def create_body_part_spatial_mask(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_ids: List[int],
    relative_patch_size: float
) -> np.ndarray:
    """Create 2D spatial mask from body part clusters (extracted from detection logic)."""
    # Combine clusters for this body part
    combined_mask = np.zeros(len(patch_components))
    for cluster_id in cluster_ids:
        if cluster_id < patch_components.shape[1]:
            combined_mask += patch_components[:, cluster_id]
    
    # Create binary patch values
    patch_values = (combined_mask > 0.1).astype(float)
    
    # Create spatial mask using existing logic
    return create_spatial_mask_from_patches(patch_coordinates, patch_values, relative_patch_size)


def calculate_centroid_from_spatial_mask(
    spatial_mask: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float
) -> Optional[Tuple[float, float]]:
    """Calculate centroid from cleaned spatial mask."""
    properties = calculate_mask_properties(spatial_mask)
    if properties['area'] == 0:
        return None
    
    # Convert spatial mask centroid back to relative coordinates
    mask_centroid = properties['centroid']
    
    # Convert from spatial mask coordinates to relative coordinates
    max_coord = np.max(patch_coordinates)
    min_coord = np.min(patch_coordinates)
    coord_range = max_coord - min_coord + relative_patch_size
    image_size = int(coord_range / relative_patch_size) + 1
    
    # Map spatial coordinates back to relative coordinates
    relative_row = (mask_centroid[0] * relative_patch_size / image_size) + min_coord
    relative_col = (mask_centroid[1] * relative_patch_size / image_size) + min_coord
    
    return (relative_row, relative_col)