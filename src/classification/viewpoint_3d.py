"""
3D Viewpoint Estimation from Body Part Distribution

Standalone functions for estimating animal viewpoint using 3D pose reconstruction.
Solves for 3D axis orientations from 2D body part centroids using orthogonality 
and body width constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from src.utils.cluster_utils import combine_cluster_patches
from src.utils.morphological_utils import apply_morphology, create_mask_from_patches, apply_mask_to_patches

# Constants - numerical tolerances only (Line 54 - Constants at top of module)
NUMERICAL_TOLERANCE = 1e-8
MAGNITUDE_TOLERANCE = 1e-6




def calculate_weighted_centroid(patch_coordinates: np.ndarray, cluster_mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate weighted centroid of patches.
    
    **Guide check**: Lines 11-15 SRP ✓ - ONLY calculates centroid
    **Guide check**: Lines 36-37 Reusability ✓ - Independently usable
    **Guide check**: Lines 123-127 Pure function ✓ - Only depends on parameters
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2)
        cluster_mask: Weights for each patch (n_patches,)
    
    Returns:
        (x, y) centroid coordinates or None if no patches
    """
    if cluster_mask.sum() == 0:
        return None
    
    weights = cluster_mask / cluster_mask.sum()
    centroid_x = np.sum(patch_coordinates[:, 0] * weights)
    centroid_y = np.sum(patch_coordinates[:, 1] * weights)
    
    return (centroid_x, centroid_y)



def classify_viewpoint_from_3d_axes(
    axes: Dict[str, Optional[np.ndarray]],
    forward_threshold: float = 45.0,
    side_threshold: float = 75.0
) -> str:
    """
    Classify viewpoint using clear angular boundaries with atan2 for numerical stability.
    
    **Guide check**: Lines 11-15 SRP ✓ - ONLY classifies viewpoint using angular approach
    **Guide check**: Lines 105 No magic numbers ✓ - Configurable angle thresholds
    **Guide check**: Lines 35-39 Reusability ✓ - Clear interpretable parameters
    
    Args:
        axes: Dict with 3D axis vectors (not required to be normalized)
        forward_threshold: Angle threshold in degrees for front/back classification
        side_threshold: Angle threshold in degrees for left/right classification
    
    Returns:
        Viewpoint classification string
    """
    if axes['x'] is None:
        return "Unknown"
    
    head_direction = axes['x']  # tail -> head direction
    
    # Project head direction onto camera XZ plane (horizontal plane)
    # This gives us the horizontal angle (azimuth) for left/right classification
    horizontal_angle = np.degrees(np.arctan2(head_direction[1], head_direction[2]))  # atan2(x, z)
    
    # Calculate vertical angle (elevation) for front/back refinement
    # horizontal_magnitude = np.sqrt(head_direction[0]**2 + head_direction[2]**2)
    # vertical_angle = np.degrees(np.arctan2(head_direction[1], horizontal_magnitude))  # atan2(y, horizontal)
    
    # Normalize horizontal angle to [0, 360) for easier interpretation
    if horizontal_angle < 0:
        horizontal_angle += 360
    
    # Classification based on horizontal angle (azimuth)
    # 0° = forward (+Z), 90° = right (+X), 180° = backward (-Z), 270° = left (-X)
    
    # Define angular ranges for each direction
    front_range = (360 - forward_threshold, forward_threshold)  # Near 180°
    right_range = (90  - side_threshold, 90  + side_threshold)  # Near 90°  
    back_range = (180 - forward_threshold, 180 + forward_threshold)  # Near 0° 
    left_range = (270 - side_threshold, 270 + side_threshold)  # Near 270°
    
    # Check which ranges the angle falls into
    is_back = (horizontal_angle <= front_range[1]) or (horizontal_angle >= front_range[0])
    is_right = right_range[0] <= horizontal_angle <= right_range[1]
    is_front = back_range[0] <= horizontal_angle <= back_range[1]
    is_left = left_range[0] <= horizontal_angle <= left_range[1]
    
    # Classification logic with clear angular boundaries
    if is_front and is_left:
        return "Front Left"
    elif is_front and is_right:
        return "Front Right"
    elif is_back and is_left:
        return "Back Left"
    elif is_back and is_right:
        return "Back Right"
    elif is_front:
        return "Front"
    elif is_back:
        return "Back"
    elif is_left:
        return "Left"
    elif is_right:
        return "Right"
    else:
        # Default to closest cardinal direction
        distances_to_cardinals = [
            abs(horizontal_angle - 180),  # Distance to front (180° )
            abs(horizontal_angle - 90 ),   # Distance to right (90°)
            min(horizontal_angle, 360 - horizontal_angle),  # Distance to back (0°)
            abs(horizontal_angle - 270)   # Distance to left (270°)
        ]
        
        closest_cardinal = np.argmin(distances_to_cardinals)
        return ["Front", "Right", "Back", "Left"][closest_cardinal]



def extract_body_part_centroids(patch_components: np.ndarray, 
                                patch_coordinates: np.ndarray, 
                                relative_patch_size: float,
                                cluster_labels: Dict[str, List[int]],
                                morphology: bool = True,
                                operations: List[str] = ["opening", "closing"],
                                kernel_size: int = 1,
                                kernel_shape: str = "disk"
                                ) -> Dict[str, Dict]:
    """
    Extract centroids and cluster masks for all labeled body parts.
    
    **Guide check**: Lines 11-15 SRP ✓ - ONLY extracts centroids
    **Guide check**: Lines 32-34 Orchestration ✓ - Uses helper functions
    
    Args:
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2)
        cluster_labels: Dict mapping body part names to cluster ID lists
    
    Returns:
        Dict with body part data: {part_name: {centroid, cluster_mask}}
    """
    body_parts = {}
    for part_name, cluster_ids in cluster_labels.items():
        cluster_mask = combine_cluster_patches(patch_components, cluster_ids)
        
        if morphology:
            mask = create_mask_from_patches(patch_coordinates, cluster_mask, relative_patch_size)
            mask = apply_morphology(mask, operations, kernel_size, kernel_shape)
            cluster_mask = apply_mask_to_patches(mask, patch_coordinates, cluster_mask, relative_patch_size)
        centroid = calculate_weighted_centroid(patch_coordinates, cluster_mask)
        if centroid is not None:
            body_parts[part_name] = {
                'centroid': centroid,
                'cluster_mask': cluster_mask
            }
    return body_parts


def fit_axis_to_combined_patches(
    body_parts: Dict[str, Dict],
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    axis_body_parts: List[str],
    image_size: Tuple[int, int]
) -> Optional[Dict]:
    """
    Fit 3D line to all patches from multiple body parts representing one anatomical axis.
    
    Style guide Lines 11-15: Single responsibility - ONLY fits axis to combined patches
    Style guide Lines 123-127: Pure function using geometric fitting
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        axis_body_parts: List of body part names to combine for this axis
        cluster_labels: Dict mapping body part names to cluster ID lists
        
    Returns:
        Dict with axis direction, centroid, and quality metrics, or None if insufficient data
    """
    # Collect all patches for this axis
    combined_mask = np.zeros(len(patch_coordinates))
    
    for part_name in axis_body_parts:
        if part_name in body_parts:
            combined_mask += body_parts[part_name]["cluster_mask"]
    
    # Clip to [0,1] and filter significant patches
    combined_mask = np.clip(combined_mask, 0, 1)
    valid_indices = combined_mask > 0.1
    
    if valid_indices.sum() < 3:
        return None
    
    # Convert valid patches to 3D points
    coords_2d = patch_coordinates[valid_indices]
    depths = patch_depths[valid_indices]
    weights = combined_mask[valid_indices]
    
    # Convert relative coordinates to pixel coordinates (preserves aspect ratio)
    width, height = image_size
    pixel_row = coords_2d[:, 0] * height  # row * height 
    pixel_col = coords_2d[:, 1] * width   # col * width
    
    # Scale depth to match pixel scale magnitude
    # max_pixel_dim = min(width, height)
    scaled_depths = depths   # Scale depth to pixel scale
    
    points_3d = np.column_stack([
        pixel_row - height/2,   # centered pixel row
        pixel_col - width/2,    # centered pixel col
        scaled_depths           # scaled depth
    ])
    
    # Weighted centroid
    centroid = np.average(points_3d, axis=0, weights=weights)
    # Weighted PCA for axis direction
    centered_points = points_3d - centroid
    weighted_points = centered_points * np.sqrt(weights[:, np.newaxis])
    try:
        # SVD to get principal direction
        _, s, Vt = np.linalg.svd(weighted_points, full_matrices=False)
        primary_direction = Vt[ 0]  # First principal component
        
        explained_variance_ratio = s[0]**2 / np.sum(s**2) if len(s) > 0 else 0
        
        return {
            'direction': primary_direction,
            'centroid': centroid,
            'quality': float(explained_variance_ratio),
            'n_patches': int(valid_indices.sum())
        }
        
    except np.linalg.LinAlgError:
        return None

def find_direction_from_body_parts(from_part: str, to_part: str,
                                   body_parts: Dict[str, Dict],
                                   image_size: Tuple[int, int]
                                   ):
    if from_part in body_parts and to_part in body_parts:
        if body_parts[from_part]['centroid'] and body_parts[to_part]['centroid']:
            to_pos = body_parts[to_part]['centroid']
            from_pos = body_parts[from_part]['centroid']
            # Convert to pixel coordinates matching fit_axis_to_combined_patches
            width, height = image_size
            to_3d = np.array([to_pos[0] * height - height/2, to_pos[1] * width - width/2, 0])
            from_3d = np.array([from_pos[0] * height - height/2, from_pos[1] * width - width/2, 0])
            direction = to_3d - from_3d
            return direction
    return None

def estimate_axes_from_patch_groups(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    image_size: Tuple[int, int],
    morphology: bool = True,
    operations: List[str] = ["opening", "closing"],
    kernel_size: int = 1,
    kernel_shape: str = "square"
) -> Dict[str, Optional[np.ndarray]]:
    """
    Estimate 3D axes by fitting lines to grouped body part patches.
    
    Style guide Lines 11-15: Single responsibility - ONLY estimates axes from patch groups
    Style guide Lines 5-9: Reuses fit_axis_to_combined_patches function
    
    Args:
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        cluster_labels: Dict mapping body part names to cluster ID lists
        
    Returns:
        Dict with 3D axes {x, y, z} and quality metrics
    """
    # Define robust axis groupings - body included in both for stability
    longitudinal_parts = ['head', 'neck', 'body', 'tail', 'thighs']
    dorsal_ventral_parts = ['legs', 'belly', 'body', 'back']  # body included for robustness
    body_parts = extract_body_part_centroids(patch_components, 
                                             patch_coordinates, 
                                             relative_patch_size,
                                             cluster_labels, 
                                             morphology, 
                                             operations, 
                                             kernel_size, 
                                             kernel_shape)
    
    # Fit X-axis (longitudinal: tail → head)
    x_axis_fit = fit_axis_to_combined_patches(
        body_parts, patch_coordinates, patch_depths,
        longitudinal_parts, image_size
    )
    
    
    # Fit Z-axis (dorsal-ventral: legs → back)
    z_axis_fit = fit_axis_to_combined_patches(
        body_parts, patch_coordinates, patch_depths,
        dorsal_ventral_parts, image_size
    )
    if z_axis_fit is None:
        z_axis_fit = {'direction': np.array([0,1,0]), 'quality': 0}
    if x_axis_fit is None:
        return {'x': None, 'y': None, 'z': None, 'x_quality': 0, 'z_quality': 0}, body_parts
    
    # Get axis directions from fitting
    x_axis = x_axis_fit['direction']
    z_axis = z_axis_fit['direction']
    
    # Get body part centroids to determine correct anatomical directions
    # print(body_parts["head"])
    # Determine correct X-axis direction from centroids (tail → head)
    x_direction_from_centroids = find_direction_from_body_parts('body', 'head', body_parts, image_size)
    if x_direction_from_centroids is None:
        x_direction_from_centroids = find_direction_from_body_parts('tail', 'body', body_parts, image_size)
    
    # Check X-axis direction using dot product (only X-Y components)
    if x_direction_from_centroids is not None:
        if np.dot(x_axis[:2], x_direction_from_centroids[:2]) < 0:
            x_axis = -x_axis
    
    # Normalize X-axis before orthogonalization
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Determine correct Z-axis direction from centroids (legs → back)
    z_direction_from_centroids = find_direction_from_body_parts('legs', 'back', body_parts, image_size)
    if z_direction_from_centroids is None:
        z_direction_from_centroids = find_direction_from_body_parts('legs', 'body', body_parts, image_size)
    if z_direction_from_centroids is None:
        z_direction_from_centroids = find_direction_from_body_parts('body', 'back', body_parts, image_size)
    if z_direction_from_centroids is None:
        z_direction_from_centroids = np.array([0,1,0])
    # Check Z-axis direction using dot product
    if z_direction_from_centroids is not None:
        if np.dot(z_axis[:2], z_direction_from_centroids[:2]) < 0:
            z_axis = -z_axis
    
    # Ensure orthogonality - orthogonalize Z with respect to X
    z_proj_on_x = np.dot(z_axis, x_axis) * x_axis
    z_axis = z_axis - z_proj_on_x
    
    # Normalize after orthogonalization
    if np.linalg.norm(z_axis) < NUMERICAL_TOLERANCE:
        return {'x': None, 'y': None, 'z': None, 'x_quality': 0, 'z_quality': 0}, body_parts
    
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Re-validate Z-axis direction after orthogonalization (legs → back)
    if z_direction_from_centroids is not None:
        if np.dot(z_axis, z_direction_from_centroids) < 0:
            z_axis = -z_axis
    
    # Y-axis from cross product (right-hand rule)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    return {
        'x': x_axis,#_fit['direction'],  # DEBUG: Return original X-axis from SVD
        'y': y_axis,
        'z': z_axis, #_fit['direction'],
        'x_quality': x_axis_fit['quality'],
        'z_quality': z_axis_fit['quality']
    }, body_parts


def estimate_viewpoint_with_axis_fitting(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    image_size: Tuple[int, int],
    forward_threshold: float = 45,
    side_threshold: float = 75,
    depth_mult: float = 50,
    morphology: bool = True,
    operations: List[str] = ["opening", "closing"],
    kernel_size: int = 1,
    kernel_shape: str = "disk"
) -> Dict:
    """
    Estimate viewpoint using robust axis fitting to grouped patches.
    
    Style guide Lines 32-34: Orchestration function - uses helper functions
    Style guide Lines 11-15: Single responsibility - ONLY orchestrates axis-fitting estimation
    
    Args:
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        cluster_labels: Dict mapping body part names to cluster ID lists
        forward_threshold: Threshold for forward/backward classification
        side_threshold: Threshold for left/right classification
        
    Returns:
        Dict with viewpoint estimation results including quality metrics
    """
    patch_depths = depth_mult * patch_depths

    # NEW: Fit axes to grouped patches
    axes_3d, body_parts = estimate_axes_from_patch_groups(
        patch_components, patch_coordinates, patch_depths, relative_patch_size, cluster_labels, image_size, 
                                             morphology, 
                                             operations, 
                                             kernel_size, 
                                             kernel_shape
    )
    
    # Use existing viewpoint classification
    viewpoint = classify_viewpoint_from_3d_axes(axes_3d, forward_threshold, side_threshold)
    
    # Calculate confidence based on axis quality and part availability
    x_quality = axes_3d.get('x_quality', 0)
    z_quality = axes_3d.get('z_quality', 0)
    available_parts = sum(1 for part in body_parts.values() if part['centroid'] is not None)
    part_coverage = available_parts / len(cluster_labels) if cluster_labels else 0.0
    
    # Combined confidence: axis quality (0.7) + part coverage (0.3)
    axis_quality = (x_quality + z_quality) / 2
    confidence = axis_quality * 0.7 + part_coverage * 0.3
    
    return {
        'viewpoint': viewpoint,
        'confidence': confidence,
        'body_parts': body_parts,
        'axes_3d': axes_3d,
        'x_quality': x_quality,
        'z_quality': z_quality,
        'axis_quality': axis_quality,
        'available_parts': available_parts,
        'part_coverage': part_coverage,
        'method': 'axis_fitting'
    }


def estimate_viewpoint_with_filtered_patches(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    image_size: Tuple[int, int],
    patch_indices: np.ndarray,
    forward_threshold: float = 45,
    side_threshold: float = 75,
    depth_mult: float = 50,
    morphology: bool = True,
    operations: List[str] = ["opening", "closing"],
    kernel_size: int = 1,
    kernel_shape: str = "square"
) -> Dict:
    """
    Estimate viewpoint using only specified patch indices from a detection.
    
    Args:
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        relative_patch_size: Size of patches relative to image
        cluster_labels: Dict mapping body part names to cluster ID lists
        image_size: (width, height) tuple
        patch_indices: Array of patch indices to use for this detection
        forward_threshold: Threshold for forward/backward classification
        side_threshold: Threshold for left/right classification
        depth_mult: Multiplier for depth values
        morphology: Whether to apply morphological operations
        operations: List of morphological operations to apply
        kernel_size: Size of morphological kernel
        kernel_shape: Shape of morphological kernel
        
    Returns:
        Dict with viewpoint estimation results for this detection
    """
    # Filter all inputs to only include the specified patch indices
    filtered_patch_components = patch_components[patch_indices]
    filtered_patch_coordinates = patch_coordinates[patch_indices]
    filtered_patch_depths = patch_depths[patch_indices] * depth_mult
    
    # Call the original function with filtered data
    return estimate_viewpoint_with_axis_fitting(
        filtered_patch_components,
        filtered_patch_coordinates, 
        filtered_patch_depths,
        relative_patch_size,
        cluster_labels,
        image_size,
        forward_threshold,
        side_threshold,
        depth_mult=1.0,  # Already applied above
        morphology=morphology,
        operations=operations,
        kernel_size=kernel_size,
        kernel_shape=kernel_shape
    )


def estimate_viewpoint_for_detections(
    detections: List[Dict],
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    image_size: Tuple[int, int],
    **viewpoint_kwargs
) -> List[Dict]:
    """
    Apply viewpoint estimation to each detection using its filtered patches.
    
    Args:
        detections: List of detection dictionaries from detect_animals_with_sam2()
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        relative_patch_size: Size of patches relative to image
        cluster_labels: Dict mapping body part names to cluster ID lists
        image_size: (width, height) tuple
        **viewpoint_kwargs: Additional arguments for viewpoint estimation
        
    Returns:
        List of detection dictionaries enriched with viewpoint information
    """
    enriched_detections = []
    
    for detection_info in detections:
        patch_indices = detection_info['patch_indices']
        
        # Estimate viewpoint for this detection's patches
        viewpoint_result = estimate_viewpoint_with_filtered_patches(
            patch_components=patch_components,
            patch_coordinates=patch_coordinates,
            patch_depths=patch_depths,
            relative_patch_size=relative_patch_size,
            cluster_labels=cluster_labels,
            image_size=image_size,
            patch_indices=patch_indices,
            **viewpoint_kwargs
        )
        
        # Add viewpoint information to detection
        enriched_detection = detection_info.copy()
        enriched_detection['viewpoint'] = viewpoint_result
        enriched_detections.append(enriched_detection)
    
    return enriched_detections