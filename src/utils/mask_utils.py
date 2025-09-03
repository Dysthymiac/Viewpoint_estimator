"""
Mask Utility Functions

Functions for creating boolean masks from patch analysis results.
Following the codebase pattern of standalone utility functions.
"""

from __future__ import annotations

from typing import List, Union, Dict
import numpy as np
from PIL import Image

from .cluster_utils import combine_cluster_patches
from ..visualization.patch_overlay import calculate_patch_bounds


def create_cluster_boolean_mask(
    image_size: tuple[int, int],
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Union[Dict[str, List[int]], List[str]],
    target_labels: List[str],
    relative_patch_size: float
) -> np.ndarray:
    """
    Create boolean mask for image pixels corresponding to specified cluster labels.
    
    Args:
        image_size: (width, height) of target image
        patch_components: Cluster membership matrix (n_patches, n_clusters) OR
                         cluster labels array (n_patches,) for direct clustering results
        patch_coordinates: Patch coordinates (n_patches, 2) with relative (row, col) positions [0,1]
        cluster_labels: Either:
                       - Dict mapping semantic labels to cluster ID lists (e.g., {"body": [0, 1], "head": [2]})
                       - List of label strings for each patch (e.g., ["body", "body", "head", ...])
        target_labels: List of cluster names to set to True (e.g., ["body", "head"])
        relative_patch_size: Relative size of each patch [0,1]
        
    Returns:
        Boolean mask array of shape (height, width) where True indicates pixels 
        belonging to the specified cluster labels
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    
    # Handle different input formats for patch_components and cluster_labels
    if isinstance(cluster_labels, dict):
        # Format: patch_components is (n_patches, n_clusters), cluster_labels is {label: [cluster_ids]}
        if patch_components.ndim != 2:
            raise ValueError("When cluster_labels is dict, patch_components must be 2D (n_patches, n_clusters)")
        
        # Combine patches from target clusters
        target_cluster_ids = []
        for label in target_labels:
            if label in cluster_labels:
                target_cluster_ids.extend(cluster_labels[label])
        
        if not target_cluster_ids:
            return mask  # Return empty mask if no target clusters found
            
        # Get combined cluster mask for target clusters
        cluster_mask = combine_cluster_patches(patch_components, target_cluster_ids)
        
    else:
        # Format: patch_components is ignored, cluster_labels is list of strings per patch
        if len(cluster_labels) != len(patch_coordinates):
            raise ValueError("Length of cluster_labels list must match number of patches")
        
        # Create boolean mask for patches with target labels
        cluster_mask = np.array([label in target_labels for label in cluster_labels], dtype=float)
    
    # Calculate patch bounds using existing function
    y_starts, x_starts, patch_height, patch_width = calculate_patch_bounds(
        patch_coordinates, image_size, relative_patch_size
    )
    
    # Set mask to True for pixels covered by target patches
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        # Only include patches with non-zero cluster membership
        if cluster_mask[i] > 0:
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            mask[y_start:y_end, x_start:x_end] = True
    
    return mask


def create_body_parts_boolean_mask(
    image_size: tuple[int, int],
    body_parts: Dict[str, Dict],
    patch_coordinates: np.ndarray,
    target_labels: List[str],
    relative_patch_size: float
) -> np.ndarray:
    """
    Create boolean mask from body_parts dictionary (as used in viewpoint estimation results).
    
    This is a convenience function for when you have a body_parts dict from viewpoint 
    estimation results rather than raw clustering data.
    
    Args:
        image_size: (width, height) of target image
        body_parts: Dict from viewpoint estimation with structure:
                   {part_name: {'centroid': (x, y), 'cluster_mask': np.ndarray}}
        patch_coordinates: Patch coordinates (n_patches, 2) with relative (row, col) positions [0,1]
        target_labels: List of body part names to set to True (e.g., ["body", "head"])
        relative_patch_size: Relative size of each patch [0,1]
        
    Returns:
        Boolean mask array of shape (height, width) where True indicates pixels
        belonging to the specified body parts
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    
    # Combine cluster masks from target body parts
    combined_cluster_mask = np.zeros(len(patch_coordinates), dtype=float)
    
    for part_name in target_labels:
        if part_name in body_parts:
            part_data = body_parts[part_name]
            if 'cluster_mask' in part_data:
                combined_cluster_mask += part_data['cluster_mask']
    
    # Clip to [0, 1] range
    combined_cluster_mask = np.clip(combined_cluster_mask, 0, 1)
    
    # Calculate patch bounds using existing function
    y_starts, x_starts, patch_height, patch_width = calculate_patch_bounds(
        patch_coordinates, image_size, relative_patch_size
    )
    
    # Set mask to True for pixels covered by target patches
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        # Only include patches with non-zero cluster membership
        if combined_cluster_mask[i] > 0:
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            mask[y_start:y_end, x_start:x_end] = True
    
    return mask


def create_threshold_boolean_mask(
    image_size: tuple[int, int],
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_indices: List[int],
    threshold: float,
    relative_patch_size: float
) -> np.ndarray:
    """
    Create boolean mask based on cluster membership threshold.
    
    Useful for creating masks from soft clustering results (e.g., Gaussian Mixture Models)
    where you want pixels with cluster membership above a certain threshold.
    
    Args:
        image_size: (width, height) of target image
        patch_components: Cluster membership matrix (n_patches, n_clusters) with values [0,1]
        patch_coordinates: Patch coordinates (n_patches, 2) with relative (row, col) positions [0,1]
        cluster_indices: List of cluster indices to consider
        threshold: Minimum cluster membership value to set pixel to True
        relative_patch_size: Relative size of each patch [0,1]
        
    Returns:
        Boolean mask array of shape (height, width) where True indicates pixels
        with cluster membership above threshold
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    
    if patch_components.ndim != 2:
        raise ValueError("patch_components must be 2D (n_patches, n_clusters)")
    
    # Combine specified cluster memberships
    combined_membership = np.zeros(len(patch_components), dtype=float)
    for cluster_idx in cluster_indices:
        if cluster_idx < patch_components.shape[1]:
            combined_membership += patch_components[:, cluster_idx]
    
    # Apply threshold
    cluster_mask = combined_membership >= threshold
    
    # Calculate patch bounds using existing function
    y_starts, x_starts, patch_height, patch_width = calculate_patch_bounds(
        patch_coordinates, image_size, relative_patch_size
    )
    
    # Set mask to True for pixels covered by patches above threshold
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        if cluster_mask[i]:
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            mask[y_start:y_end, x_start:x_end] = True
    
    return mask