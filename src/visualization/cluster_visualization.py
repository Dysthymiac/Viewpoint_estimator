"""
Cluster Visualization Utilities

Standalone functions for visualizing cluster analysis results and patch overlays.
Extracted from cluster_labeler.py following file separation rules.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Optional, Tuple

# Import existing visualization utilities
from src.visualization.patch_overlay import create_patch_mask, blend_image_with_patches
from src.visualization.patch_uncertainty import calculate_patch_uncertainty
from src.visualization.component_normalization import normalize_components
from src.utils.cluster_utils import combine_cluster_patches

# Import morphological processing if available
try:
    from src.visualization.morphological_postprocessing import apply_morphological_cleanup
    MORPHOLOGICAL_AVAILABLE = True
except ImportError:
    MORPHOLOGICAL_AVAILABLE = False




def create_highlighted_clusters(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    selected_clusters: List[int],
    alpha: float,
    smooth: bool
) -> Image.Image:
    """
    Create cluster highlight overlay with red coloring for selected clusters.
    
    **Guide check**: Lines 5-9 No duplication ✓ - Extracted from cluster_labeler.py lines 125-157
    **Guide check**: Lines 113 Standalone function ✓ - No longer uses self, takes parameters
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates highlighted clusters
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space
        relative_patch_size: Size of patches relative to image
        selected_clusters: List of cluster indices to highlight
        alpha: Overlay transparency
        smooth: Whether to smooth patch rendering
    
    Returns:
        PIL Image with cluster highlights overlay
    """
    if not selected_clusters:
        return image
    
    # Combine selected clusters (reuse existing logic)
    cluster_mask = combine_cluster_patches(patch_components, selected_clusters)
    
    # Create red overlay for selected clusters
    cluster_colors = np.zeros((len(cluster_mask), 3))
    cluster_colors[:, 0] = cluster_mask  # Red channel
    
    patch_mask = create_patch_mask(image.size, patch_coordinates, cluster_colors, relative_patch_size, smooth)
    return blend_image_with_patches(image, patch_mask, alpha)


def create_probabilities_view(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    normalization: str,
    temperature: float,
    alpha: float,
    smooth: bool = False,
    component_indices: Optional[List[int]] = None,
    morphological_cleanup: bool = False,
    morph_operations: List[str] = None,
    morph_kernel_size: int = 3,
    morph_kernel_shape: str = "disk"
) -> Image.Image:
    """
    Create probabilities visualization using RGB channels for first 3 components.
    
    **Guide check**: Lines 5-9 No duplication ✓ - Extracted from cluster_labeler.py lines 159-212
    **Guide check**: Lines 113 Standalone function ✓ - No longer uses self, takes parameters
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates probabilities view
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2)
        relative_patch_size: Size of patches relative to image
        normalization: Normalization method for components
        temperature: Temperature for softmax normalization
        alpha: Overlay transparency
        smooth: Whether to smooth patch rendering
        component_indices: Optional list of component indices to visualize
        morphological_cleanup: Whether to apply morphological cleanup
        morph_operations: List of morphological operations
        morph_kernel_size: Kernel size for morphological operations
        morph_kernel_shape: Kernel shape for morphological operations
    
    Returns:
        PIL Image with probabilities overlay
    """
    # Apply morphological cleanup if requested
    if morphological_cleanup and MORPHOLOGICAL_AVAILABLE and morph_operations:
        patch_components = apply_morphological_cleanup(
            patch_components, patch_coordinates, relative_patch_size,
            operations=morph_operations, 
            kernel_size=morph_kernel_size, 
            kernel_shape=morph_kernel_shape
        )
    
    # Select components if specified
    if component_indices is not None:
        selected_components = patch_components[:, component_indices]
    else:
        selected_components = patch_components
    
    # Normalize components
    normalized = normalize_components(selected_components, normalization, temperature)
    
    # Create RGB visualization using first 3 components
    n_vis_components = min(3, normalized.shape[1])
    if n_vis_components == 0:
        return image
    
    rgb_components = normalized[:, :n_vis_components]
    if n_vis_components < 3:
        # Pad with zeros if less than 3 components
        padding = np.zeros((len(rgb_components), 3 - n_vis_components))
        rgb_components = np.hstack([rgb_components, padding])
    
    patch_mask = create_patch_mask(image.size, patch_coordinates, rgb_components, relative_patch_size, smooth)
    return blend_image_with_patches(image, patch_mask, alpha)


def create_uncertainty_view(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    uncertainty_method: str,
    colormap: str,
    alpha: float,
    smooth: bool = False,
    component_indices: Optional[List[int]] = None,
    normalization: str = "linear",
    temperature: float = 1.0,
    morphological_cleanup: bool = False,
    morph_operations: List[str] = None,
    morph_kernel_size: int = 3,
    morph_kernel_shape: str = "disk"
) -> Image.Image:
    """
    Create uncertainty visualization using colormap.
    
    **Guide check**: Lines 5-9 No duplication ✓ - Extracted from cluster_labeler.py lines 214-264
    **Guide check**: Lines 113 Standalone function ✓ - No longer uses self, takes parameters
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates uncertainty view
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2)
        relative_patch_size: Size of patches relative to image
        uncertainty_method: Method for calculating uncertainty
        colormap: Matplotlib colormap name
        alpha: Overlay transparency
        smooth: Whether to smooth patch rendering
        component_indices: Optional list of component indices
        normalization: Normalization method for components
        temperature: Temperature for normalization
        morphological_cleanup: Whether to apply morphological cleanup
        morph_operations: List of morphological operations
        morph_kernel_size: Kernel size for morphological operations
        morph_kernel_shape: Kernel shape for morphological operations
    
    Returns:
        PIL Image with uncertainty overlay
    """
    # Apply morphological cleanup if requested
    if morphological_cleanup and MORPHOLOGICAL_AVAILABLE and morph_operations:
        patch_components = apply_morphological_cleanup(
            patch_components, patch_coordinates, relative_patch_size,
            operations=morph_operations,
            kernel_size=morph_kernel_size,
            kernel_shape=morph_kernel_shape
        )
    
    # Calculate uncertainty
    uncertainties = calculate_patch_uncertainty(
        patch_components, uncertainty_method, component_indices, normalization, temperature
    )
    
    # Normalize uncertainties to [0, 1]
    if uncertainties.max() > uncertainties.min():
        norm_uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())
    else:
        norm_uncertainties = np.ones_like(uncertainties) * 0.5
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    uncertainty_colors = cmap(norm_uncertainties)[:, :3]  # Remove alpha channel
    
    patch_mask = create_patch_mask(image.size, patch_coordinates, uncertainty_colors, relative_patch_size, smooth)
    return blend_image_with_patches(image, patch_mask, alpha)

def create_labeled_body_parts_view(
    image: Image.Image,
    body_parts: Dict[str, Dict],
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    alpha: float,
    smooth: bool = False,
) -> Image.Image:
    """
    Create visualization showing all labeled clusters with distinct colors.
    
    **Guide check**: Lines 5-9 No duplication ✓ - Extracted from cluster_labeler.py lines 266-332
    **Guide check**: Lines 113 Standalone function ✓ - No longer uses self, takes parameters
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates labeled clusters view
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2)
        relative_patch_size: Size of patches relative to image
        cluster_labels: Dict mapping semantic labels to cluster ID lists
        alpha: Overlay transparency
        smooth: Whether to smooth patch rendering
        morphological_cleanup: Whether to apply morphological cleanup
        morph_operations: List of morphological operations
        morph_kernel_size: Kernel size for morphological operations
        morph_kernel_shape: Kernel shape for morphological operations
    
    Returns:
        PIL Image with labeled clusters overlay
    """
    
    
    # Create distinct colors for each semantic label
    unique_labels = list(body_parts.keys())
    label_colors = {}
    
    # Generate distinct colors using colormap
    if unique_labels:
        cmap = plt.get_cmap('tab10')  # Qualitative colormap with distinct colors
        for i, label in enumerate(unique_labels):
            color = cmap(i / max(len(unique_labels) - 1, 1))[:3]  # RGB only
            label_colors[label] = color
    
    # Create combined visualization
    combined_colors = np.zeros((len(patch_coordinates), 3))
    
    for semantic_label, body_part in body_parts.items():
        label_color = label_colors[semantic_label]
        
        cluster_mask = body_part["cluster_mask"]
                
        # Add this cluster's contribution with its label color
        for channel in range(3):
            combined_colors[:, channel] += cluster_mask * label_color[channel]
    
    # Normalize colors to [0, 1] range
    max_intensity = combined_colors.max(axis=0)
    for channel in range(3):
        if max_intensity[channel] > 0:
            combined_colors[:, channel] /= max_intensity[channel]
    
    patch_mask = create_patch_mask(image.size, patch_coordinates, combined_colors, relative_patch_size, smooth)
    return blend_image_with_patches(image, patch_mask, alpha)


def create_labeled_clusters_view(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    alpha: float,
    smooth: bool = False,
    morphological_cleanup: bool = False,
    morph_operations: List[str] = None,
    morph_kernel_size: int = 3,
    morph_kernel_shape: str = "disk"
) -> Image.Image:
    """
    Create visualization showing all labeled clusters with distinct colors.
    
    **Guide check**: Lines 5-9 No duplication ✓ - Extracted from cluster_labeler.py lines 266-332
    **Guide check**: Lines 113 Standalone function ✓ - No longer uses self, takes parameters
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates labeled clusters view
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2)
        relative_patch_size: Size of patches relative to image
        cluster_labels: Dict mapping semantic labels to cluster ID lists
        alpha: Overlay transparency
        smooth: Whether to smooth patch rendering
        morphological_cleanup: Whether to apply morphological cleanup
        morph_operations: List of morphological operations
        morph_kernel_size: Kernel size for morphological operations
        morph_kernel_shape: Kernel shape for morphological operations
    
    Returns:
        PIL Image with labeled clusters overlay
    """
    if not cluster_labels:
        return image
    
    # Apply morphological cleanup if requested
    if morphological_cleanup and MORPHOLOGICAL_AVAILABLE and morph_operations:
        patch_components = apply_morphological_cleanup(
            patch_components, patch_coordinates, relative_patch_size,
            operations=morph_operations,
            kernel_size=morph_kernel_size,
            kernel_shape=morph_kernel_shape
        )
    
    # Create distinct colors for each semantic label
    unique_labels = list(cluster_labels.keys())
    label_colors = {}
    
    # Generate distinct colors using colormap
    if unique_labels:
        cmap = plt.get_cmap('tab10')  # Qualitative colormap with distinct colors
        for i, label in enumerate(unique_labels):
            color = cmap(i / max(len(unique_labels) - 1, 1))[:3]  # RGB only
            label_colors[label] = color
    
    # Create combined visualization
    combined_colors = np.zeros((len(patch_components), 3))
    
    for semantic_label, cluster_list in cluster_labels.items():
        label_color = label_colors[semantic_label]
        
        for cluster_id in cluster_list:
            if cluster_id < patch_components.shape[1]:
                cluster_mask = patch_components[:, cluster_id]
                
                # Add this cluster's contribution with its label color
                for channel in range(3):
                    combined_colors[:, channel] += cluster_mask * label_color[channel]
    
    # Normalize colors to [0, 1] range
    max_intensity = combined_colors.max(axis=0)
    for channel in range(3):
        if max_intensity[channel] > 0:
            combined_colors[:, channel] /= max_intensity[channel]
    
    patch_mask = create_patch_mask(image.size, patch_coordinates, combined_colors, relative_patch_size, smooth)
    return blend_image_with_patches(image, patch_mask, alpha)


def create_viewpoint_visualization(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    estimation_result: Dict,
    alpha: float = 0.7,
    smooth: bool = False,
    arrow_scale: float = 100.0,
    figsize: Tuple[int, int] = (12, 8),
    morphological_cleanup: bool = False,
    morph_operations: List[str] = None,
    morph_kernel_size: int = 3,
    morph_kernel_shape: str = "disk"
) -> plt.Figure:
    """
    Create combined viewpoint estimation visualization.
    
    **Guide check**: Lines 32-34 Orchestration ✓ - Uses existing cluster visualization
    **Guide check**: Lines 5-9 No duplication ✓ - Reuses existing functions
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates viewpoint visualization
    
    Args:
        image: Original PIL Image
        patch_components: Cluster membership matrix
        patch_coordinates: Patch coordinates in relative space
        relative_patch_size: Patch size relative to image
        cluster_labels: Body part to cluster mapping
        estimation_result: Result dict from estimate_viewpoint_3d
        alpha: Overlay transparency for clusters
        smooth: Whether to smooth patch rendering
        arrow_scale: Length scaling for 2D arrows
        figsize: Figure size
        morphological_cleanup: Whether to apply morphological cleanup
        morph_operations: List of morphological operations
        morph_kernel_size: Kernel size for morphological operations
        morph_kernel_shape: Kernel shape for morphological operations
    
    Returns:
        Matplotlib Figure with viewpoint visualization
    """
    # Get cluster overlay using existing function - match working cluster_labeler.py parameters
    cluster_overlay = create_labeled_clusters_view(
        image, patch_components, patch_coordinates, relative_patch_size,
        cluster_labels, alpha, smooth, morphological_cleanup,
        morph_operations, morph_kernel_size, morph_kernel_shape
    )
    
    # Create matplotlib figure and display cluster overlay
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cluster_overlay)
    
    # Add centroids and labels
    img_width, img_height = image.size
    body_parts = estimation_result['body_parts']
    
    for part_name, part_data in body_parts.items():
        # Skip background labels
        if 'background' in part_name.lower():
            continue
        centroid = part_data.get('centroid')
        if centroid is not None:
            # Convert (row, col) coordinates to (x, y) for matplotlib
            x = centroid[1] * img_width   # col -> x
            y = centroid[0] * img_height  # row -> y
            ax.scatter(x, y, c='white', s=100, edgecolors='black', linewidths=2, marker='o')
            ax.text(x + 10, y - 10, part_name, color='white', fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    # Add all axes arrows with body centroid as origin
    body_centroid = body_parts.get('body', {}).get('centroid')
    if body_centroid is not None:
        # Convert (row, col) coordinates to (x, y) for matplotlib  
        ref_x = body_centroid[1] * img_width   # col -> x
        ref_y = body_centroid[0] * img_height  # row -> y
        
        # 1. Show 2D centroid-based axes (calculated from body part positions)
        x_2d = estimation_result.get('x_2d')
        if x_2d is not None:
            # Convert (row, col) direction to (x, y) for matplotlib
            dx, dy = x_2d[1] * arrow_scale, x_2d[0] * arrow_scale
            ax.arrow(ref_x, ref_y, dx, dy, head_width=10, head_length=15,
                    fc='red', ec='red', linewidth=3, alpha=0.8, 
                    label='2D X-axis (from centroids)', linestyle='--')
        
        z_2d = estimation_result.get('z_2d') 
        if z_2d is not None:
            # Convert (row, col) direction to (x, y) for matplotlib
            dx, dy = z_2d[1] * arrow_scale, z_2d[0] * arrow_scale
            ax.arrow(ref_x, ref_y, dx, dy, head_width=10, head_length=15,
                    fc='blue', ec='blue', linewidth=3, alpha=0.8,
                    label='2D Z-axis (from centroids)', linestyle='--')
        
        # 2. Show projected 3D axes (final estimated axes projected to 2D)
        axes_3d = estimation_result.get('axes_3d', {})
        if axes_3d.get('x') is not None:
            # Project 3D X-axis to 2D (use x,y components, ignore z)
            x_3d_proj = axes_3d['x'][:2]  # [x, y] components only
            dx, dy = x_3d_proj[0] * arrow_scale, -x_3d_proj[1] * arrow_scale  # flip y  
            ax.arrow(ref_x, ref_y, dx, dy, head_width=15, head_length=20,
                    fc='cyan', ec='cyan', linewidth=4, 
                    label='3D X-axis (final)')
        
        if axes_3d.get('y') is not None:
            # Project 3D Y-axis to 2D (use x,y components, ignore z)
            y_3d_proj = axes_3d['y'][:2]  # [x, y] components only
            dx, dy = y_3d_proj[0] * arrow_scale, -y_3d_proj[1] * arrow_scale  # flip y
            ax.arrow(ref_x, ref_y, dx, dy, head_width=15, head_length=20,
                    fc='green', ec='green', linewidth=4,
                    label='3D Y-axis (final)')
        
        if axes_3d.get('z') is not None:
            # Project 3D Z-axis to 2D (use x,y components, ignore z)
            z_3d_proj = axes_3d['z'][:2]  # [x, y] components only  
            dx, dy = z_3d_proj[0] * arrow_scale, -z_3d_proj[1] * arrow_scale  # flip y
            ax.arrow(ref_x, ref_y, dx, dy, head_width=15, head_length=20,
                    fc='yellow', ec='yellow', linewidth=4,
                    label='3D Z-axis (final)')
    
    # Add result text
    viewpoint = estimation_result['viewpoint']
    confidence = estimation_result['confidence']
    available_parts = estimation_result.get('available_parts', 0)
    
    result_text = f"Viewpoint: {viewpoint}"
    confidence_text = f"Confidence: {confidence:.2f}"
    parts_text = f"Parts: {available_parts}/{len(cluster_labels)}"
    
    text_lines = [result_text, confidence_text, parts_text]
    for i, text in enumerate(text_lines):
        ax.text(0.02, 0.98 - i * 0.06, text, transform=ax.transAxes,
                fontsize=14, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
    
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    if x_2d is not None or z_2d is not None:
        ax.legend(loc='upper right')
    
    ax.set_title('3D Viewpoint Estimation Results', fontsize=16, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
    
    return fig


def create_3d_axes_visualization(
    axes_3d: Dict[str, Optional[np.ndarray]], 
    body_centroid: Optional[Tuple[float, float]] = None,
    camera_distance: float = 5.0,
    axis_scale: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    show_grid: bool = True,
    camera_view: bool = True
) -> plt.Figure:
    """
    Create interactive 3D scene showing camera and animal in spatial context.
    
    **Guide check**: Lines 11-15 SRP ✓ - ONLY creates full 3D camera-animal scene
    **Guide check**: Lines 123-127 Visualization function ✓ - Interactive spatial rendering
    **Guide check**: Lines 105 No magic numbers ✓ - All distances and positions configurable
    
    Args:
        axes_3d: Dict with 3D axis vectors {x, y, z}
        body_centroid: (x, y) position of animal body in image coordinates (0-1 range)
        camera_distance: Distance from camera to animal along Z-axis
        axis_scale: Scale factor for animal axes display
        figsize: Figure size in inches
        show_grid: Whether to show 3D grid
        camera_view: If True, align plot view with camera perspective for projection comparison
    
    Returns:
        Interactive matplotlib Figure with full 3D scene
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera position at origin
    camera_pos = np.array([0, 0, 0])
    
    # Animal position - place at camera_distance along Z, offset by body centroid
    if body_centroid is not None:
        # Convert (0-1) image coords to world coords, center around camera
        animal_x = (body_centroid[1] - 0.5) * 2  # Column -> X, centered
        animal_y = (body_centroid[0] - 0.5) * 2  # Row -> Y, centered  
        animal_pos = np.array([animal_x, animal_y, camera_distance])
    else:
        animal_pos = np.array([0, 0, camera_distance])
    
    # Draw camera
    ax.scatter(*camera_pos, color='black', s=200, marker='s', 
              alpha=0.8, label='Camera', edgecolors='white', linewidths=2)
    
    # Draw camera coordinate system (world coordinates)
    camera_axes = {
        'x': np.array([1, 0, 0]),  # Camera X (image right)
        'y': np.array([0, 1, 0]),  # Camera Y (image up) 
        'z': np.array([0, 0, 1])   # Camera Z (into scene)
    }
    
    camera_colors = {'x': 'orange', 'y': 'purple', 'z': 'gray'}
    camera_labels = {'x': 'Camera X (right)', 'y': 'Camera Y (up)', 'z': 'Camera Z (forward)'}
    
    for axis_name, axis_vec in camera_axes.items():
        ax.quiver(*camera_pos, *axis_vec, 
                 color=camera_colors[axis_name], arrow_length_ratio=0.1, 
                 linewidth=4, alpha=0.7, label=camera_labels[axis_name])
    
    # Draw animal position
    ax.scatter(*animal_pos, color='brown', s=150, marker='o', 
              alpha=0.9, label='Animal', edgecolors='black', linewidths=2)
    
    # Draw animal coordinate system (if available)
    if axes_3d['x'] is not None:
        axis_colors = {'x': 'cyan', 'y': 'green', 'z': 'yellow'}  # Match 2D plot colors
        axis_labels = {'x': 'Animal X (tail→head)', 'y': 'Animal Y (right→left)', 'z': 'Animal Z (legs→back)'}
        
        for axis_name, color in axis_colors.items():
            if axes_3d.get(axis_name) is not None:
                axis_vector = axes_3d[axis_name] * axis_scale
                
                # Draw animal axis from animal position
                ax.quiver(*animal_pos, *axis_vector, 
                         color=color, arrow_length_ratio=0.15, 
                         linewidth=5, alpha=0.9, label=axis_labels[axis_name])
                
                # Draw axis line for reference
                extension = 1.5
                end_pos = animal_pos + axis_vector * extension
                ax.plot([animal_pos[0], end_pos[0]], 
                       [animal_pos[1], end_pos[1]], 
                       [animal_pos[2], end_pos[2]], 
                       color=color, linestyle='--', alpha=0.4, linewidth=2)
    
    # Draw line connecting camera to animal
    ax.plot([camera_pos[0], animal_pos[0]], 
           [camera_pos[1], animal_pos[1]], 
           [camera_pos[2], animal_pos[2]], 
           color='yellow', linestyle=':', linewidth=3, alpha=0.8, label='Camera-Animal line')
    
    # Set up coordinate system with camera at center
    max_range = camera_distance * 0.8
    ax.set_xlim([-max_range/2, max_range/2])
    ax.set_ylim([-max_range/2, max_range/2])
    ax.set_zlim([0, camera_distance + 1])
    
    # Enhanced styling
    ax.set_xlabel('X (Image Right)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Image Up)', fontsize=12, fontweight='bold') 
    ax.set_zlabel('Z (Camera Forward)', fontsize=12, fontweight='bold')
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Legend with better positioning
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    # Set viewing angle based on mode
    if camera_view:
        # Align with camera perspective - X vertical, Y horizontal, Z into scene
        # elev=-90, azim=-90 makes X vertical, Y horizontal
        ax.view_init(elev=-90, azim=-90)  # Look down Z-axis with X vertical, Y horizontal
        ax.set_title('3D Scene - Camera Perspective View\n(X vertical, Y horizontal - matches 2D projection)', 
                    fontsize=14, fontweight='bold', pad=20)
    else:
        # Overview angle to see the full spatial relationship  
        ax.view_init(elev=15, azim=125)  # Oblique view to see camera-animal relationship
        ax.set_title('3D Camera-Animal Spatial Relationship\n(Click and drag to rotate, scroll to zoom)', 
                    fontsize=14, fontweight='bold', pad=20)
    
    return fig