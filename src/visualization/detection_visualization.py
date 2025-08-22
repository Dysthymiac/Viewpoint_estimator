"""
Animal Detection Visualization Functions

Standalone visualization functions for animal detection results.
Following style guide Lines 49-57: separate visualization concerns from core algorithm.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
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