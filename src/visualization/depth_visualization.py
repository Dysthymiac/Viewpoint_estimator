"""
Depth Visualization Utilities

Standalone functions for visualizing DINOv2 depth estimation results.
Following style guide Lines 50-54: separate visualization concerns into dedicated modules.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
from scipy.interpolate import griddata

# Import existing visualization utilities - Style guide Lines 5-9: reuse existing functions
from .patch_overlay import create_patch_mask, blend_image_with_patches
from .cluster_visualization import create_labeled_clusters_view, create_labeled_body_parts_view


def normalize_depth_values(depth_values: np.ndarray) -> np.ndarray:
    """
    Normalize depth values to [0, 1] range for colormap application.
    
    Style guide Lines 11-15: Single responsibility - ONLY normalizes depth values
    Style guide Lines 123-127: Pure function that only depends on parameters
    
    Args:
        depth_values: Raw depth values from DINOv2 depth head
        
    Returns:
        Normalized depth values in [0, 1] range
    """
    if depth_values.max() > depth_values.min():
        normalized = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
    else:
        # Handle case where all values are the same
        normalized = np.ones_like(depth_values) * 0.5
    
    return normalized


def apply_depth_colormap(depth_values: np.ndarray, colormap: str = 'viridis') -> np.ndarray:
    """
    Apply matplotlib colormap to normalized depth values.
    
    Style guide Lines 11-15: Single responsibility - ONLY applies colormap
    Pattern follows create_uncertainty_view Lines 227-231 in cluster_visualization.py
    
    Args:
        depth_values: Normalized depth values in [0, 1] range
        colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'inferno', 'turbo')
        
    Returns:
        RGB colors array of shape (n_patches, 3)
    """
    cmap = plt.get_cmap(colormap)
    depth_colors = cmap(depth_values)[:, :3]  # Remove alpha channel
    return depth_colors


def create_depth_overlay(
    image: Image.Image,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    colormap: str = 'viridis',
    alpha: float = 0.7,
    smooth: bool = False
) -> Image.Image:
    """
    Create depth visualization overlay on original image.
    
    Style guide Lines 5-9: Reuses existing create_patch_mask and blend_image_with_patches
    Style guide Lines 11-15: Single responsibility - ONLY creates depth overlay
    Pattern follows create_uncertainty_view in cluster_visualization.py
    
    Args:
        image: Original PIL Image
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Raw depth values from DINOv2 depth head (n_patches,)
        relative_patch_size: Size of patches relative to image dimensions
        colormap: Matplotlib colormap name for depth visualization
        alpha: Overlay transparency (0=original image, 1=depth only)
        smooth: Whether to apply bilinear interpolation for smooth heatmap
        
    Returns:
        PIL Image with depth overlay
    """
    # Normalize depth values
    normalized_depths = normalize_depth_values(patch_depths)
    
    # Apply colormap
    depth_colors = apply_depth_colormap(normalized_depths, colormap)
    
    # Create patch mask using existing utility - Style guide Lines 5-9: no duplication
    patch_mask = create_patch_mask(
        image.size, patch_coordinates, depth_colors, relative_patch_size, smooth
    )
    
    # Blend with original image using existing utility
    return blend_image_with_patches(image, patch_mask, alpha)


def create_depth_visualization(
    image: Image.Image,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    relative_patch_size: float,
    colormap: str = 'viridis',
    alpha: float = 0.7,
    smooth: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    show_stats: bool = True
) -> plt.Figure:
    """
    Create complete depth visualization with statistics and colorbar.
    
    Style guide Lines 32-34: Orchestration function - uses existing depth overlay
    Style guide Lines 11-15: Single responsibility - ONLY creates complete depth visualization
    Pattern follows create_viewpoint_visualization in cluster_visualization.py
    
    Args:
        image: Original PIL Image
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Raw depth values from DINOv2 depth head (n_patches,)
        relative_patch_size: Size of patches relative to image dimensions
        colormap: Matplotlib colormap name for depth visualization
        alpha: Overlay transparency
        smooth: Whether to apply bilinear interpolation
        figsize: Figure size in inches
        show_stats: Whether to show depth statistics on plot
        
    Returns:
        Matplotlib Figure with depth visualization and colorbar
    """
    # Create depth overlay using existing function - Style guide Lines 5-9: reuse
    depth_overlay = create_depth_overlay(
        image, patch_coordinates, patch_depths, relative_patch_size, 
        colormap, alpha, smooth
    )
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(depth_overlay)
    
    # Add colorbar
    normalized_depths = normalize_depth_values(patch_depths)
    cmap = plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=patch_depths.min(), vmax=patch_depths.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Depth', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Add statistics if requested
    if show_stats:
        depth_min = patch_depths.min()
        depth_max = patch_depths.max()
        depth_mean = patch_depths.mean()
        depth_std = patch_depths.std()
        
        stats_text = f"Depth Range: {depth_min:.3f} to {depth_max:.3f}\\n"
        stats_text += f"Mean: {depth_mean:.3f} ± {depth_std:.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=12, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
                verticalalignment='top')
    
    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title('DINOv2 Depth Estimation Results', fontsize=16, color='white', weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8))
    
    return fig


def interpolate_depth_surface(
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    grid_resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate sparse patch depths to regular grid for 3D surface visualization.
    
    Style guide Lines 11-15: Single responsibility - ONLY interpolates depth surface
    Style guide Lines 123-127: Pure function that only depends on parameters
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Raw depth values from DINOv2 depth head (n_patches,)
        grid_resolution: Resolution of output grid (grid_resolution x grid_resolution)
        
    Returns:
        Tuple of (X_grid, Y_grid, Z_grid) for 3D surface plotting
    """
    # Create regular grid in relative coordinates [0,1]
    x_grid = np.linspace(0, 1, grid_resolution)
    y_grid = np.linspace(0, 1, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate depth values to grid using cubic interpolation
    # Note: patch_coordinates are (row, col) but we need (x, y) for griddata
    points = np.column_stack([patch_coordinates[:, 1], patch_coordinates[:, 0]])  # (col, row) -> (x, y)
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Use linear interpolation for robustness (cubic can be unstable with sparse data)
    Z_grid = griddata(points, patch_depths, grid_points, method='linear', fill_value=patch_depths.mean())
    Z_grid = Z_grid.reshape(X_grid.shape)
    
    return X_grid, Y_grid, Z_grid


def create_3d_depth_surface(
    image: Image.Image,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    grid_resolution: int = 50,
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (14, 10),
    elevation: float = 30,
    azimuth: float = 45,
    show_wireframe: bool = False,
    show_original_points: bool = True,
    invert_depth: bool = True
) -> plt.Figure:
    """
    Create 3D surface visualization of depth estimation.
    
    Style guide Lines 11-15: Single responsibility - ONLY creates 3D depth surface
    Style guide Lines 5-9: Reuses interpolate_depth_surface function
    Pattern follows create_3d_axes_visualization in cluster_visualization.py
    
    Args:
        image: Original PIL Image (for aspect ratio and statistics)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Raw depth values from DINOv2 depth head (n_patches,)
        grid_resolution: Resolution of interpolated surface grid
        colormap: Matplotlib colormap name
        figsize: Figure size in inches
        elevation: 3D plot elevation angle in degrees
        azimuth: 3D plot azimuth angle in degrees  
        show_wireframe: Whether to show wireframe overlay
        show_original_points: Whether to show original patch points as scatter
        invert_depth: If True, invert depth values (closer = higher in plot)
        
    Returns:
        Matplotlib Figure with 3D depth surface
    """
    # Interpolate depth surface using existing function - Style guide Lines 5-9: reuse
    X_grid, Y_grid, Z_grid = interpolate_depth_surface(patch_coordinates, patch_depths, grid_resolution)
    
    # Optionally invert depth for better visualization (closer objects appear higher)
    if invert_depth:
        Z_grid = -Z_grid
        plot_depths = -patch_depths
        depth_label = 'Inverted Depth (closer = higher)'
    else:
        plot_depths = patch_depths
        depth_label = 'Depth'
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surface = ax.plot_surface(
        X_grid, Y_grid, Z_grid, 
        cmap=colormap, alpha=0.9, 
        linewidth=0, antialiased=True,
        rasterized=True  # For better rendering performance
    )
    
    # Add wireframe if requested
    if show_wireframe:
        ax.plot_wireframe(X_grid, Y_grid, Z_grid, color='black', alpha=0.3, linewidth=0.5)
    
    # Show original patch points if requested
    if show_original_points:
        # Convert coordinates for 3D plot: (row, col) -> (x, y)
        patch_x = patch_coordinates[:, 1]  # col -> x
        patch_y = patch_coordinates[:, 0]  # row -> y
        ax.scatter(patch_x, patch_y, plot_depths, c='red', s=20, alpha=0.8, label='Original patches')
    
    # Add colorbar
    cbar = plt.colorbar(surface, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(depth_label, rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X (Image Width)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Image Height)', fontsize=12, fontweight='bold')
    ax.set_zlabel(depth_label, fontsize=12, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add statistics
    depth_min = patch_depths.min()
    depth_max = patch_depths.max()
    depth_mean = patch_depths.mean()
    depth_std = patch_depths.std()
    depth_range = depth_max - depth_min
    
    stats_text = f"Depth Statistics:\\n"
    stats_text += f"Range: {depth_min:.3f} to {depth_max:.3f}\\n"
    stats_text += f"Mean: {depth_mean:.3f} ± {depth_std:.3f}\\n"
    stats_text += f"Dynamic Range: {depth_range:.3f}"
    
    # Add text box with statistics
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
              fontsize=10, color='black', weight='bold',
              bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
              verticalalignment='top')
    
    # Add image info
    img_info = f"Image: {image.size[0]}×{image.size[1]}\\n"
    img_info += f"Patches: {len(patch_coordinates)}\\n" 
    img_info += f"Grid: {grid_resolution}×{grid_resolution}"
    
    ax.text2D(0.98, 0.98, img_info, transform=ax.transAxes,
              fontsize=10, color='black', weight='bold',
              bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
              verticalalignment='top', horizontalalignment='right')
    
    if show_original_points:
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))
    
    ax.set_title('3D Depth Surface from DINOv2\\n(Interactive: click and drag to rotate, scroll to zoom)', 
                fontsize=14, fontweight='bold', pad=20)
    
    return fig


def create_depth_cross_section(
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    image: Image.Image,
    cross_section_type: str = 'horizontal',
    position: float = 0.5,
    grid_resolution: int = 100,
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create cross-section view of depth estimation along horizontal or vertical line.
    
    Style guide Lines 11-15: Single responsibility - ONLY creates depth cross-section
    Style guide Lines 5-9: Reuses interpolate_depth_surface for consistency
    
    Args:
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Raw depth values from DINOv2 depth head (n_patches,)
        image: Original PIL Image (for reference)
        cross_section_type: 'horizontal' or 'vertical' cross-section
        position: Position of cross-section line in [0,1] range
        grid_resolution: Resolution for interpolation along cross-section
        colormap: Matplotlib colormap name
        figsize: Figure size in inches
        
    Returns:
        Matplotlib Figure with depth cross-section plot
    """
    # Get interpolated surface for cross-section extraction
    X_grid, Y_grid, Z_grid = interpolate_depth_surface(patch_coordinates, patch_depths, grid_resolution)
    
    # Extract cross-section
    if cross_section_type == 'horizontal':
        # Horizontal line at specified Y position
        y_index = int(position * (grid_resolution - 1))
        x_line = X_grid[y_index, :]
        depth_line = Z_grid[y_index, :]
        line_label = f"Horizontal cross-section at Y={position:.2f}"
        x_label = "X (Image Width)"
    else:  # vertical
        # Vertical line at specified X position
        x_index = int(position * (grid_resolution - 1))
        x_line = Y_grid[:, x_index]
        depth_line = Z_grid[:, x_index]
        line_label = f"Vertical cross-section at X={position:.2f}"
        x_label = "Y (Image Height)"
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Top plot: cross-section line
    ax1.plot(x_line, depth_line, linewidth=3, color='blue', label=line_label)
    ax1.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Depth', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Add statistics
    depth_min = patch_depths.min()
    depth_max = patch_depths.max()
    ax1.axhline(depth_min, color='green', linestyle='--', alpha=0.7, label=f'Min: {depth_min:.3f}')
    ax1.axhline(depth_max, color='red', linestyle='--', alpha=0.7, label=f'Max: {depth_max:.3f}')
    ax1.legend()
    
    # Bottom plot: show cross-section position on depth map
    depth_colors = apply_depth_colormap(normalize_depth_values(patch_depths), colormap)
    
    # Create 2D depth visualization for reference
    X_2d, Y_2d, Z_2d = interpolate_depth_surface(patch_coordinates, patch_depths, 50)
    im = ax2.contourf(X_2d, Y_2d, Z_2d, levels=20, cmap=colormap, alpha=0.8)
    
    # Show cross-section line
    if cross_section_type == 'horizontal':
        ax2.axhline(position, color='white', linewidth=3, linestyle='-', label=f'Cross-section at Y={position:.2f}')
    else:
        ax2.axvline(position, color='white', linewidth=3, linestyle='-', label=f'Cross-section at X={position:.2f}')
    
    ax2.scatter(patch_coordinates[:, 1], patch_coordinates[:, 0], c='red', s=10, alpha=0.6, label='Original patches')
    ax2.set_xlabel('X (Image Width)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (Image Height)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.legend()
    
    # Add colorbar for bottom plot
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Depth', rotation=270, labelpad=15, fontsize=10)
    
    plt.suptitle(f'Depth Cross-Section Analysis\\nImage: {image.size[0]}×{image.size[1]}, Patches: {len(patch_coordinates)}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def visualize_depth_estimated_axes(
    image: Image.Image,
    estimation_result: Dict,
    patch_coordinates: np.ndarray,
    patch_depths: np.ndarray,
    patch_components: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    figsize: Tuple[int, int] = (15, 10),
    axis_scale: float = 100.0,
    alpha: float = 0.7,
    show_3d_view: bool = True
) -> plt.Figure:
    """
    Visualize depth-estimated 3D axes overlaid on body parts visualization.
    
    Style guide Lines 11-15: Single responsibility - ONLY visualizes depth-estimated axes
    Style guide Lines 5-9: Reuses existing create_labeled_clusters_view function
    
    Args:
        image: Original PIL Image
        estimation_result: Result from estimate_viewpoint_with_axis_fitting
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_depths: Depth values for each patch (n_patches,)
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        cluster_labels: Dict mapping body part names to cluster ID lists
        relative_patch_size: Size of patches relative to image dimensions
        figsize: Figure size in inches
        axis_scale: Length scaling for axis arrows in pixels
        show_3d_view: Whether to show additional 3D perspective view
        
    Returns:
        Matplotlib Figure with axis visualization
    """
    # Create figure with subplots
    if show_3d_view:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
        ax2 = None
    
    # Get axes and body parts
    axes_3d = estimation_result.get('axes_3d', {})
    body_parts = estimation_result.get('body_parts', {})
    
    # Find body centroid as reference point
    body_centroid = None
    if 'body' in body_parts and body_parts['body']['centroid'] is not None:
        body_centroid = body_parts['body']['centroid']
    else:
        # Use center of image as fallback
        body_centroid = (0.5, 0.5)
    
    # Create body parts overlay as background
    body_parts_overlay = create_labeled_body_parts_view(
        image, body_parts, patch_coordinates, relative_patch_size,
        alpha=alpha, smooth=False
    )
    
    # Plot 1: 2D view with projected axes
    ax1.imshow(body_parts_overlay)
    
    # Convert body centroid to pixel coordinates
    img_width, img_height = image.size
    ref_x = body_centroid[1] * img_width   # col -> x
    ref_y = body_centroid[0] * img_height  # row -> y
    
    # Draw 3D axes projected to 2D
    axis_colors = {'x': 'cyan', 'y': 'lime', 'z': 'yellow'}
    axis_labels = {'x': 'X-axis (tail→head)', 'y': 'Y-axis (left→right)', 'z': 'Z-axis (legs→back)'}
    
    for axis_name, color in axis_colors.items():
        if axes_3d.get(axis_name) is not None:
            axis_3d = axes_3d[axis_name]
            
            # Project 3D axis to 2D: [row, col, depth] -> [col, row] for display
            dx = axis_3d[1] * axis_scale   # col component -> image x
            dy = axis_3d[0] * axis_scale   # row component -> image y
            # Draw arrow
            ax1.arrow(ref_x, ref_y, dx, dy, 
                     head_width=15, head_length=20, fc=color, ec=color, 
                     linewidth=4, alpha=0.9, label=axis_labels[axis_name])
    
    # Add body part centroids
    for part_name, part_data in body_parts.items():
        # Skip background labels
        if 'background' in part_name.lower():
            continue
        centroid = part_data.get('centroid')
        if centroid is not None:
            x_pixel = centroid[1] * img_width
            y_pixel = centroid[0] * img_height
            ax1.scatter(x_pixel, y_pixel, c='white', s=80, edgecolors='black', 
                       linewidths=2, marker='o', alpha=0.8)
            ax1.text(x_pixel + 10, y_pixel - 10, part_name, color='white', 
                    fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    # Add estimation results text
    viewpoint = estimation_result.get('viewpoint', 'Unknown')
    confidence = estimation_result.get('confidence', 0.0)
    x_quality = estimation_result.get('x_quality', 0.0)
    z_quality = estimation_result.get('z_quality', 0.0)
    
    result_text = f"Viewpoint: {viewpoint}\n"
    result_text += f"Confidence: {confidence:.3f}\n"
    result_text += f"X-axis quality: {x_quality:.3f}\n"
    result_text += f"Z-axis quality: {z_quality:.3f}"
    
    ax1.text(0.02, 0.98, result_text, transform=ax1.transAxes,
             fontsize=12, color='white', weight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
             verticalalignment='top')
    
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)  # Flip y-axis for image coordinates
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_title('Depth-Estimated 3D Axes (2D Projection)', fontsize=14, weight='bold')
    
    # Plot 2: 3D perspective view (if requested)
    if show_3d_view and ax2 is not None:
        ax2.remove()  # Remove 2D axis
        ax2 = fig.add_subplot(212, projection='3d')
        
        # Convert body centroid to 3D position using average depth
        if 'body' in body_parts and body_parts['body']['cluster_mask'] is not None:
            body_mask = body_parts['body']['cluster_mask']
            if body_mask.sum() > 0:
                weights = body_mask / body_mask.sum()
                body_depth = np.sum(patch_depths * weights)
            else:
                body_depth = patch_depths.mean()
        else:
            body_depth = patch_depths.mean()
        
        # 3D reference point - switch x and y for correct camera alignment
        img_width, img_height = image.size
        ref_3d = np.array([
            body_centroid[1] * img_width - img_width/2,    # col -> x 
            body_centroid[0] * img_height - img_height/2,  # row -> y 
            body_depth
        ])
        
        # Draw coordinate system origin
        ax2.scatter(*ref_3d, color='red', s=100, marker='o', 
                   alpha=0.8, label='Body center')
        
        # Draw 3D axes
        axis_scale_3d = 0.3  # Scale for 3D view
        for axis_name, color in axis_colors.items():
            if axes_3d.get(axis_name) is not None:
                # Switch x and y components for camera alignment
                axis_calc = axes_3d[axis_name]
                axis_3d = np.array([
                    axis_calc[1],  # col -> x
                    axis_calc[0],  # row -> y  
                    axis_calc[2]   # depth -> z
                ]) * axis_scale_3d
                
                # Draw axis vector
                ax2.quiver(*ref_3d, *axis_3d, color=color, 
                          arrow_length_ratio=0.15, linewidth=4, 
                          alpha=0.9, label=axis_labels[axis_name])
        
        # Set 3D view properties
        ax2.set_xlabel('X (Image Width)', fontsize=10, weight='bold')
        ax2.set_ylabel('Y (Image Height)', fontsize=10, weight='bold')
        ax2.set_zlabel('Z (Depth)', fontsize=10, weight='bold')
        
        # Set equal aspect ratio for 3D
        max_range = axis_scale_3d * 1.5
        ax2.set_xlim([ref_3d[0] - max_range, ref_3d[0] + max_range])
        ax2.set_ylim([ref_3d[1] - max_range, ref_3d[1] + max_range])
        ax2.set_zlim([ref_3d[2] - max_range, ref_3d[2] + max_range])
        
        # Set viewing angle aligned with camera view
        # For [row, col, depth] coordinate system to match camera:
        # - row should be vertical (Y in plot)
        # - col should be horizontal (X in plot) 
        # - depth should point into screen (Z in plot)
        ax2.view_init(elev=90, azim=-90)
        ax2.invert_yaxis()  # Invert Y-axis to match image coordinates
        ax2.invert_zaxis()
        ax2.legend(loc='upper left', fontsize=9)
        ax2.set_title('3D Axes in Camera Space', fontsize=14, weight='bold')
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig