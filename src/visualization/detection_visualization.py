"""
Animal Detection Visualization Functions

Standalone visualization functions for animal detection results.
Following style guide Lines 49-57: separate visualization concerns from core algorithm.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import label

from ..detection.animal_detector import (
    AnimalDetection,
    collect_animal_clusters,
    create_animal_mask,
    extract_detections
)


def plot_animal_detections(image: Image.Image, detections: List[AnimalDetection]) -> 'plt.Figure':
    """
    Plot animal detections - returns matplotlib figure.
    
    Style guide Lines 49-57: Visualization separated from core algorithm
    Style guide Lines 11-15: SRP ✓ - ONLY plots detections
    
    Args:
        image: Original image
        detections: List of animal detections
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')
    
    for i, detection in enumerate(detections):
        add_detection_to_plot(ax, detection, image.size, i)
    
    return fig


def add_detection_to_plot(ax: 'plt.Axes', detection: AnimalDetection, image_size: tuple, index: int) -> None:
    """
    Add single detection to plot axis.
    
    Style guide Lines 11-15: SRP ✓ - ONLY adds detection to plot
    Style guide Lines 49-57: Visualization separated from core algorithm
    """
    
    width, height = image_size
    bbox = detection.bbox
    
    # Convert to absolute coordinates
    x1 = bbox.x1 * width
    y1 = bbox.y1 * height
    box_width = (bbox.x2 - bbox.x1) * width
    box_height = (bbox.y2 - bbox.y1) * height
    
    # Add bounding box
    rect = patches.Rectangle(
        (x1, y1), box_width, box_height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        x1, y1 - 5, f'Animal {index+1}: {detection.confidence:.2f}',
        color='red', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )


def debug_animal_detection_pipeline(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    kernel_size: int = 5,
    image_path: str = None
) -> Tuple[List[AnimalDetection], Dict]:
    """
    Debug every step of animal detection with visualizations.
    
    Style guide Lines 49-57: Debug/visualization function separated from core algorithm
    Style guide Lines 11-15: SRP ✓ - ONLY debugs and visualizes detection pipeline
    """
    
    print(f"Debug: {image_path or 'Unknown image'}")
    print(f"Shapes: patches={patch_components.shape}, coords={patch_coordinates.shape}")
    
    # Step 1: Debug cluster collection
    animal_clusters = collect_animal_clusters(cluster_labels, background_labels)
    print(f"Animal clusters: {animal_clusters}")
    
    # Step 2: Create mask
    active_patches = sum(np.sum(patch_components[:, c] > 0) for c in animal_clusters if c < patch_components.shape[1])
    print(f"Active patches: {active_patches}")
    
    try:
        final_mask = create_animal_mask(patch_components, patch_coordinates, animal_clusters, relative_patch_size)
        print(f"Mask: {final_mask.shape}, pixels: {np.sum(final_mask)}")
    except Exception as e:
        print(f"ERROR: {e}")
        return [], {'error': str(e)}
    
    # Step 3: Apply morphology
    opened_mask = binary_opening(final_mask, disk(kernel_size))
    cleaned_mask = binary_closing(opened_mask, disk(kernel_size))
    print(f"Morphology: {np.sum(final_mask)} → {np.sum(cleaned_mask)} pixels")
    
    # Step 4: Extract regions
    labeled_mask, num_regions = label(cleaned_mask)
    print(f"Regions: {num_regions}")
    
    # Step 5: Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Show input patch data (sum across animal clusters)
    combined_patch_mask = np.zeros(len(patch_components))
    for cluster_id in animal_clusters:
        if cluster_id < patch_components.shape[1]:
            combined_patch_mask += patch_components[:, cluster_id]
    
    # Show patch data in a reasonable grid for visualization
    try:
        # Try square grid first
        grid_size = int(np.sqrt(len(combined_patch_mask)))
        if grid_size * grid_size == len(combined_patch_mask):
            patch_mask_2d = combined_patch_mask.reshape(grid_size, grid_size)
        else:
            # Use rectangular grid
            rows = min(50, len(combined_patch_mask))  # Max 50 rows
            cols = len(combined_patch_mask) // rows
            if rows * cols < len(combined_patch_mask):
                cols += 1
            # Pad if necessary
            padded_mask = np.pad(combined_patch_mask, (0, rows * cols - len(combined_patch_mask)), 'constant')
            patch_mask_2d = padded_mask[:rows * cols].reshape(rows, cols)
    except Exception as e:
        # Fallback: just show as 1D
        patch_mask_2d = combined_patch_mask.reshape(1, -1)
    
    axes[0,0].imshow(patch_mask_2d, cmap='viridis')
    axes[0,0].set_title('Animal Clusters')
    
    axes[0,1].imshow(final_mask, cmap='gray')
    axes[0,1].set_title('Spatial Grid')
    
    axes[0,2].imshow(opened_mask, cmap='gray')
    axes[0,2].set_title('After Opening')
    
    axes[1,0].imshow(cleaned_mask, cmap='gray')
    axes[1,0].set_title('After Closing')
    
    axes[1,1].imshow(labeled_mask, cmap='tab10')
    axes[1,1].set_title(f'{num_regions} Regions')
    
    # Show final detections
    detections = extract_detections(cleaned_mask)
    detection_overlay = cleaned_mask.astype(float)
    axes[1,2].imshow(detection_overlay, cmap='gray')
    axes[1,2].set_title(f'{len(detections)} Detections')
    
    # Add bounding boxes
    for i, detection in enumerate(detections):
        bbox = detection.bbox
        h, w = cleaned_mask.shape
        rect = patches.Rectangle(
            (bbox.x1 * w, bbox.y1 * h), 
            (bbox.x2 - bbox.x1) * w, 
            (bbox.y2 - bbox.y1) * h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[1,2].add_patch(rect)
        axes[1,2].text(bbox.x1 * w, bbox.y1 * h - 5, f'{detection.confidence:.2f}', 
                      color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    debug_info = {
        'animal_clusters': animal_clusters,
        'spatial_mask': final_mask,
        'opened_mask': opened_mask, 
        'cleaned_mask': cleaned_mask,
        'labeled_mask': labeled_mask,
        'num_regions': num_regions
    }
    
    print(f"Results: {len(detections)} detections")
    
    return detections, debug_info


def visualize_sam2_detections(
    image: Image.Image,
    detections: List[Dict],
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    show_patches: bool = True,
    figsize: Tuple[int, int] = (15, 10)
) -> 'plt.Figure':
    """
    Visualize SAM2 detections showing masks and optionally patches.
    
    Args:
        image: Input image as PIL Image
        detections: List of detection dictionaries from detect_animals_with_sam2()
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        relative_patch_size: Size of patches relative to image
        show_patches: Whether to show individual patches for each detection
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure
    """
    n_detections = len(detections)
    if n_detections == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        ax.set_title('No detections found')
        ax.axis('off')
        return fig
    
    # Create subplots: original image + one subplot per detection
    cols = min(3, n_detections)
    rows = (n_detections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols > 1:
        axes = axes.reshape(1, -1) if n_detections > 0 else [axes]
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Original image
    # axes_flat[0].imshow(image)
    # axes_flat[0].set_title('Original Image')
    # axes_flat[0].axis('off')
    
    # Colors for different detections
    colors = plt.cm.tab10(np.linspace(0, 1, n_detections))
    w, h = image.size  # PIL Image uses (width, height)
    
    # Show each detection
    for i, detection_info in enumerate(detections):
        ax = axes_flat[i]
        mask = detection_info['detection'].mask
        if mask.shape != (h, w):
            image = image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS)
            (w, h) = image.size
        ax.imshow(image)
        
        # Get detection data
        patch_indices = detection_info['patch_indices']
        sam2_score = detection_info['sam2_score']
        body_part = detection_info.get('body_part', 'unknown')
        
        # Show SAM2 mask (upsample to match image size if needed)
        # if mask.shape != (h, w):
        #     from PIL import Image as PILImage
        #     mask_pil = PILImage.fromarray(mask.astype(np.uint8))
        #     mask = np.array(mask_pil.resize((w, h), PILImage.NEAREST)).astype(bool)
        
        mask_colored = np.zeros((*mask.shape, 4))
        mask_colored[mask > 0] = [*colors[i][:3], 0.6]
        ax.imshow(mask_colored)
        
        # Show body part bounding box if available
        if 'body_part_box' in detection_info and detection_info['body_part_box'] is not None:
            bbox = detection_info['body_part_box']
            
            # Convert relative coordinates to pixels
            x1_px = bbox.x1 * w
            y1_px = bbox.y1 * h
            box_width = (bbox.x2 - bbox.x1) * w
            box_height = (bbox.y2 - bbox.y1) * h
            
            # Add bounding box rectangle
            bbox_rect = patches.Rectangle(
                (x1_px, y1_px), box_width, box_height,
                linewidth=2, edgecolor='lime', facecolor='none', alpha=0.9
            )
            ax.add_patch(bbox_rect)
        
        # Show patches if requested
        if show_patches and len(patch_indices) > 0:
            patch_coords = patch_coordinates[patch_indices]
            
            for coord in patch_coords:
                y_rel, x_rel = coord
                y_px = int(y_rel * h)
                x_px = int(x_rel * w)
                
                # Calculate patch bounds
                patch_h = max(1, int(relative_patch_size * max(h, w)))
                patch_w = patch_h
                
                rect = patches.Rectangle(
                    (x_px, y_px), patch_w, patch_h,
                    linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)
        
        ax.set_title(f'Detection {i+1}: {body_part}\nScore: {sam2_score:.2f}, Patches: {len(patch_indices)}')
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(n_detections + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_detection_viewpoints(
    image: Image.Image,
    detections_with_viewpoints: List[Dict],
    patch_coordinates: np.ndarray,
    patch_components: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    image_resize: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 15)
) -> 'plt.Figure':
    """
    Visualize detections with viewpoint information showing body parts, centroids and axes.
    
    Args:
        image: Input image as PIL Image
        detections_with_viewpoints: List of enriched detection dictionaries with viewpoint info
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        cluster_labels: Dict mapping body part names to cluster ID lists
        relative_patch_size: Size of patches relative to image
        image_resize: Optional image resize parameter
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure
    """
    if image_resize is not None:
        original_width, original_height = image.size
        if original_width >= original_height:
            new_width = image_resize
            new_height = int(original_height * image_resize / original_width)
        else: 
            new_height = image_resize
            new_width = int(original_width * image_resize / original_height)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
    n_detections = len(detections_with_viewpoints)
    if n_detections == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        ax.set_title('No detections with viewpoints found')
        ax.axis('off')
        return fig
    
    # Create grid layout
    cols = min(2, n_detections)
    rows = (n_detections + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 and cols > 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    w, h = image.size  # PIL Image uses (width, height)
    
    for i, detection_info in enumerate(detections_with_viewpoints):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        ax.imshow(image)
        
        # Get detection and viewpoint data
        detection = detection_info['detection']
        viewpoint_result = detection_info['viewpoint']
        patch_indices = detection_info['patch_indices']
        
        mask = detection.mask
        viewpoint = viewpoint_result['viewpoint']
        confidence = viewpoint_result['confidence']
        body_parts = viewpoint_result['body_parts']
        axes_3d = viewpoint_result['axes_3d']
        
        # Show detection mask (upsample to match image size if needed)
        if mask.shape != (h, w):
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask.astype(np.uint8))
            mask = np.array(mask_pil.resize((w, h), PILImage.NEAREST)).astype(bool)
        
        mask_colored = np.zeros((*mask.shape, 4))
        mask_colored[mask > 0] = [0.8, 0.8, 0.8, 0.4]  # Gray mask
        ax.imshow(mask_colored)
        
        # Show body part centroids
        colors = {'head': 'red', 'body': 'blue', 'tail': 'green', 'legs': 'orange', 'back': 'purple'}
        origin_x = None
        for part_name, part_data in body_parts.items():
            if part_name == "background":
                continue
            if part_data['centroid'] is not None:
                centroid = part_data['centroid']
                cent_x = centroid[1] * w  # centroid is (row, col) -> (y, x)
                cent_y = centroid[0] * h
                if part_name == "body" or origin_x is None:
                    origin_x, origin_y = (cent_x, cent_y)
                color = colors.get(part_name, 'white')
                ax.scatter(cent_x, cent_y, c=color, s=100, marker='o', edgecolors='black', linewidth=2)
                ax.text(cent_x + 10, cent_y, part_name, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Show 3D axes projected onto image
        if axes_3d['x'] is not None:
            # Use image center as origin for axis visualization
            # origin_x, origin_y = w // 2, h // 2
            
            # Scale axes for visualization (make them visible but not too large)
            axis_scale = min(w, h) * 0.2
            
            if axes_3d['x'] is not None:
                x_axis = axes_3d['x']
                # Project 3D axis to 2D (use x and y components, ignore z for 2D projection)
                end_x = origin_x + x_axis[1] * axis_scale  # x_axis[1] is the horizontal component
                end_y = origin_y + x_axis[0] * axis_scale  # x_axis[0] is the vertical component
                ax.annotate('', xy=(end_x, end_y), xytext=(origin_x, origin_y),
                           arrowprops=dict(arrowstyle='->', color='red', lw=3))
                ax.text(end_x, end_y, 'X (head→tail)', color='red', fontweight='bold')
            
            if axes_3d['y'] is not None:
                y_axis = axes_3d['y']
                end_x = origin_x + y_axis[1] * axis_scale
                end_y = origin_y + y_axis[0] * axis_scale
                ax.annotate('', xy=(end_x, end_y), xytext=(origin_x, origin_y),
                           arrowprops=dict(arrowstyle='->', color='green', lw=3))
                ax.text(end_x, end_y, 'Y (side)', color='green', fontweight='bold')
            
            if axes_3d['z'] is not None:
                z_axis = axes_3d['z']
                end_x = origin_x + z_axis[1] * axis_scale
                end_y = origin_y + z_axis[0] * axis_scale
                ax.annotate('', xy=(end_x, end_y), xytext=(origin_x, origin_y),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=3))
                ax.text(end_x, end_y, 'Z (up/down)', color='blue', fontweight='bold')
        
        colors = {'head': 'red', 
                  'body': 'blue', 
                  'tail': 'green', 
                  'legs': 'orange', 
                  'back': 'purple', 
                  'belly': 'pink',
                  'neck': 'magenta',
                  'thighs': 'yellow'}
        # Show patches colored by body part
        if len(patch_indices) > 0:
            # Get patch components for this detection
            detection_patch_components = patch_components[patch_indices]
            patch_coords = patch_coordinates[patch_indices]
            
            # Color patches according to their dominant body part
            for i, coord in enumerate(patch_coords):
                y_rel, x_rel = coord
                y_px = int(y_rel * h)
                x_px = int(x_rel * w)
                
                patch_h = max(1, int(relative_patch_size * max(w, h)))
                patch_w = patch_h
                
                # Find the dominant body part for this patch
                patch_color = 'gray'  # Default color
                max_membership = 0.1  # Threshold for membership
                
                for part_name, cluster_ids in cluster_labels.items():
                    if part_name == 'background':
                        continue
                    
                    for cluster_id in cluster_ids:
                        if cluster_id < detection_patch_components.shape[1]:
                            membership = detection_patch_components[i, cluster_id]
                            if membership > max_membership:
                                max_membership = membership
                                patch_color = colors.get(part_name, 'white')
                
                # Create colored patch rectangle
                rect = patches.Rectangle(
                    (x_px, y_px), patch_w, patch_h,
                    linewidth=0.5, edgecolor='black', facecolor=patch_color, alpha=0.7
                )
                ax.add_patch(rect)
        
        ax.set_title(f'Detection {i+1}: {viewpoint}\nConfidence: {confidence:.2f}, Patches: {len(patch_indices)}')
        ax.axis('off')
    
    # Hide unused subplots
    for j in range(n_detections, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    return fig