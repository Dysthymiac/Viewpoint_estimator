"""
Detection Utility Functions

Standalone utility functions for processing detection-related data.
Following style guide Lines 36-40: Reusable utility functions.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..data.cvat_loader import BoundingBox


def filter_patches_for_detection(
    patch_coordinates: np.ndarray,
    patch_components: np.ndarray,
    patch_depths: Optional[np.ndarray],
    detection_bbox: BoundingBox,
    relative_patch_size: float
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Filter patches that fall within a specific detection bounding box.
    
    Style guide Lines 5-9: No duplication ✓ - Centralized patch filtering utility
    Style guide Lines 11-15: SRP ✓ - ONLY filters patches by bounding box
    Style guide Lines 36-40: Reusability ✓ - Used by multiple detection scenarios
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_depths: Optional depth values for each patch (n_patches,)
        detection_bbox: BoundingBox in relative coordinates [0,1]
        relative_patch_size: Size of patches relative to image dimensions [0,1]
        
    Returns:
        Tuple of (filtered_coordinates, filtered_components, filtered_depths)
        - filtered_coordinates: (n_filtered_patches, 2)
        - filtered_components: (n_filtered_patches, n_clusters)  
        - filtered_depths: (n_filtered_patches,) or None if input was None
    """
    # Calculate patch bounds for each patch
    half_patch_size = relative_patch_size / 2
    
    # Patch bounds in relative coordinates [0,1]
    patch_x1 = patch_coordinates[:, 1] - half_patch_size  # col - half_size
    patch_y1 = patch_coordinates[:, 0] - half_patch_size  # row - half_size  
    patch_x2 = patch_coordinates[:, 1] + half_patch_size  # col + half_size
    patch_y2 = patch_coordinates[:, 0] + half_patch_size  # row + half_size
    
    # Detection bbox bounds in relative coordinates [0,1]
    det_x1, det_y1, det_x2, det_y2 = detection_bbox.x1, detection_bbox.y1, detection_bbox.x2, detection_bbox.y2
    
    # Find patches that overlap with detection bbox
    # A patch overlaps if: patch_x1 < det_x2 AND patch_x2 > det_x1 AND patch_y1 < det_y2 AND patch_y2 > det_y1
    overlap_mask = (
        (patch_x1 < det_x2) & 
        (patch_x2 > det_x1) & 
        (patch_y1 < det_y2) & 
        (patch_y2 > det_y1)
    )
    
    # Filter patches based on overlap
    filtered_coordinates = patch_coordinates[overlap_mask]
    filtered_components = patch_components[overlap_mask]
    
    # Filter depths if provided
    filtered_depths = None
    if patch_depths is not None:
        filtered_depths = patch_depths[overlap_mask]
    
    return filtered_coordinates, filtered_components, filtered_depths


def calculate_detection_coverage(
    patch_coordinates: np.ndarray,
    detection_bbox: BoundingBox,
    relative_patch_size: float
) -> float:
    """
    Calculate what fraction of patches fall within a detection bounding box.
    
    Style guide Lines 11-15: SRP ✓ - ONLY calculates coverage metric
    Style guide Lines 36-40: Reusability ✓ - Useful for detection quality assessment
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        detection_bbox: BoundingBox in relative coordinates [0,1]
        relative_patch_size: Size of patches relative to image dimensions [0,1]
        
    Returns:
        Coverage fraction between 0.0 and 1.0
    """
    if len(patch_coordinates) == 0:
        return 0.0
    
    filtered_coords, _, _ = filter_patches_for_detection(
        patch_coordinates, 
        np.zeros((len(patch_coordinates), 1)),  # Dummy components
        None,  # No depths needed
        detection_bbox,
        relative_patch_size
    )
    
    return len(filtered_coords) / len(patch_coordinates)


def validate_detection_patches(
    patch_coordinates: np.ndarray,
    patch_components: np.ndarray,
    detection_bbox: BoundingBox,
    relative_patch_size: float,
    min_patches: int = 10
) -> bool:
    """
    Validate that a detection has sufficient patches for reliable viewpoint estimation.
    
    Style guide Lines 11-15: SRP ✓ - ONLY validates detection patch count
    Style guide Lines 36-40: Reusability ✓ - Used for quality control
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2)
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        detection_bbox: BoundingBox in relative coordinates [0,1]
        relative_patch_size: Size of patches relative to image dimensions
        min_patches: Minimum number of patches required for reliable estimation
        
    Returns:
        True if detection has sufficient patches, False otherwise
    """
    filtered_coords, filtered_components, _ = filter_patches_for_detection(
        patch_coordinates, patch_components, None, detection_bbox, relative_patch_size
    )
    
    # Check if we have enough patches
    if len(filtered_coords) < min_patches:
        return False
    
    # Check if we have meaningful cluster activation (not all zeros)
    total_activation = np.sum(filtered_components)
    if total_activation < 0.1:  # Very low activation threshold
        return False
    
    return True