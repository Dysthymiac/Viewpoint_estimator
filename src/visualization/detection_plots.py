"""Simple detection visualization utilities following style guide principles."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from ..detection.animal_detector import AnimalDetection
from ..data.cvat_loader import BoundingBox


def plot_image_with_detections(image: Image.Image, detections: List[AnimalDetection]) -> plt.Figure:
    """Plot image with detection overlays - returns matplotlib figure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')
    
    for i, detection in enumerate(detections):
        add_detection_bbox(ax, detection.bbox, image.size, f'Animal {i+1}', detection.confidence)
    
    return fig


def add_detection_bbox(ax: plt.Axes, bbox: BoundingBox, image_size: tuple, label: str, confidence: float) -> None:
    """Add single detection bounding box to axis."""
    width, height = image_size
    
    # Convert relative to absolute coordinates
    x1 = bbox.x1 * width
    y1 = bbox.y1 * height
    box_width = (bbox.x2 - bbox.x1) * width
    box_height = (bbox.y2 - bbox.y1) * height
    
    # Add rectangle
    rect = patches.Rectangle(
        (x1, y1), box_width, box_height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        x1, y1 - 5, f'{label}: {confidence:.2f}',
        color='red', fontsize=12, weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )


def add_mask_overlay(ax: plt.Axes, mask: np.ndarray, color: str = 'red', alpha: float = 0.3) -> None:
    """Add mask overlay to axis."""
    height, width = mask.shape
    overlay = np.zeros((height, width, 4))
    
    if color == 'red':
        overlay[:, :, 0] = mask
    elif color == 'green':
        overlay[:, :, 1] = mask
    elif color == 'blue':
        overlay[:, :, 2] = mask
    
    overlay[:, :, 3] = mask * alpha
    ax.imshow(overlay)


def plot_detection_statistics(detection_results: List[dict]) -> plt.Figure:
    """Plot detection performance statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # IoU distribution
    ious = [r['evaluation']['best_iou'] for r in detection_results]
    ax1.hist(ious, bins=20, alpha=0.7, color='skyblue')
    ax1.set_xlabel('IoU')
    ax1.set_ylabel('Count')
    ax1.set_title('Detection IoU Distribution')
    ax1.axvline(0.5, color='red', linestyle='--', label='IoU = 0.5')
    ax1.legend()
    
    # Detection counts
    detection_counts = [len(r['detections']) for r in detection_results]
    ax2.hist(detection_counts, bins=range(max(detection_counts) + 2), alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Number of Detections')
    ax2.set_ylabel('Count')
    ax2.set_title('Detections per Image')
    
    plt.tight_layout()
    return fig


def plot_viewpoint_scores(viewpoint_scores: dict, estimated_viewpoint: str) -> plt.Figure:
    """Plot viewpoint estimation scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    viewpoints = list(viewpoint_scores.keys())
    scores = list(viewpoint_scores.values())
    
    colors = ['red' if vp == estimated_viewpoint else 'skyblue' for vp in viewpoints]
    bars = ax.bar(viewpoints, scores, color=colors)
    
    ax.set_ylabel('Score')
    ax.set_title('Viewpoint Estimation Scores')
    ax.set_ylim(0, 1)
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        weight = 'bold' if viewpoints[i] == estimated_viewpoint else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', weight=weight)
    
    return fig


def plot_body_part_centroids(body_part_analysis: dict, title: str = "Body Part Centroids") -> plt.Figure:
    """Plot body part centroid positions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract present body parts
    present_parts = {}
    for part, analysis in body_part_analysis.items():
        if isinstance(analysis, dict) and analysis.get('present', False):
            present_parts[part] = (
                analysis.get('centroid_x', 0.5),
                analysis.get('centroid_y', 0.5)
            )
    
    if not present_parts:
        ax.text(0.5, 0.5, 'No body parts detected', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Plot centroids
    colors = plt.cm.Set3(np.linspace(0, 1, len(present_parts)))
    for i, (part, (x, y)) in enumerate(present_parts.items()):
        ax.scatter(x, y, c=[colors[i]], s=100, label=part)
        ax.annotate(part, (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X Position (relative)')
    ax.set_ylabel('Y Position (relative)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis to match image coordinates
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_spatial_analysis(spatial_analysis: dict) -> plt.Figure:
    """Plot spatial analysis metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    metrics = ['overall_aspect_ratio', 'axis_ratio', 'spread_x', 'spread_y']
    titles = ['Aspect Ratio', 'Axis Ratio', 'X Spread', 'Y Spread']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        value = spatial_analysis.get(metric, 0)
        axes[i].bar([title], [value], color='lightblue')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'{title}: {value:.3f}')
    
    plt.tight_layout()
    return fig