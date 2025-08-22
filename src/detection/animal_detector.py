"""Animal detection using labeled clusters and morphological processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..data.cvat_loader import BoundingBox
from ..utils.morphological_utils import (
    calculate_mask_properties,
    create_continuous_regions
)
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import label


@dataclass
class AnimalDetection:
    """Container for single animal detection result."""
    bbox: BoundingBox
    mask: np.ndarray
    confidence: float
    area_ratio: float
    centroid: Tuple[float, float]


def collect_animal_clusters(cluster_labels: Dict[str, List[int]], background_labels: List[str]) -> List[int]:
    """Extract cluster indices for animal body parts (non-background).
    
    Args:
        cluster_labels: Dictionary mapping semantic labels to cluster indices
        background_labels: List of labels to exclude (e.g., 'background', 'vegetation')
        
    Returns:
        List of cluster indices for animal body parts
    """
    return [c for label, clusters in cluster_labels.items() if label not in background_labels for c in clusters]


def create_animal_mask(patch_components: np.ndarray, patch_coordinates: np.ndarray, 
                      animal_clusters: List[int], relative_patch_size: float) -> np.ndarray:
    """Create binary spatial mask from animal patch clusters.
    
    Args:
        patch_components: Shape (n_patches, n_components) with cluster assignments
        patch_coordinates: Shape (n_patches, 2) with relative coordinates [0,1]
        animal_clusters: List of cluster indices to include
        relative_patch_size: Relative size of each patch [0,1]
        
    Returns:
        Binary mask with animal regions marked
    """
    import math
    
    # Combine animal clusters
    combined_mask = np.zeros(len(patch_components))
    for cluster_id in animal_clusters:
        if cluster_id < patch_components.shape[1]:
            combined_mask += patch_components[:, cluster_id]
    
    # Create binary patch values
    patch_values = (combined_mask > 0.1).astype(float)
    
    # Calculate image bounds from coordinate range
    max_coord = np.max(patch_coordinates)
    min_coord = np.min(patch_coordinates)
    coord_range = max_coord - min_coord + relative_patch_size
    image_size = int(coord_range / relative_patch_size) + 1
    
    # Initialize spatial mask
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Calculate patch size in pixels
    patch_height = math.ceil(relative_patch_size * image_size)
    patch_width = math.ceil(relative_patch_size * image_size)
    
    # Convert coordinates to pixel positions
    y_starts = np.round((patch_coordinates[:, 0] - min_coord) / relative_patch_size).astype(int)
    x_starts = np.round((patch_coordinates[:, 1] - min_coord) / relative_patch_size).astype(int)
    
    # Ensure bounds
    y_starts = np.clip(y_starts, 0, image_size - patch_height)
    x_starts = np.clip(x_starts, 0, image_size - patch_width)
    
    # Fill rectangular patch areas with binary values
    for i, (y_start, x_start) in enumerate(zip(y_starts, x_starts)):
        if patch_values[i] > 0:
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            mask[y_start:y_end, x_start:x_end] = patch_values[i]
    
    return (mask > 0.5).astype(np.uint8)


def apply_morphology(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Clean binary mask using morphological opening and closing.
    
    Args:
        mask: Binary input mask
        kernel_size: Size of circular structuring element
        
    Returns:
        Cleaned binary mask
    """
    kernel = disk(kernel_size)
    cleaned_mask = binary_opening(mask, kernel)
    return binary_closing(cleaned_mask, kernel)


def get_component_bbox(component_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate tight bounding box around binary component.
    
    Args:
        component_mask: Binary mask of connected component
        
    Returns:
        Tuple of (min_row, min_col, max_row, max_col)
    """
    rows, cols = np.where(component_mask)
    return (rows.min(), cols.min(), rows.max(), cols.max())


def create_detection_from_component(component_mask: np.ndarray, bbox: Tuple[int, int, int, int], 
                                  mask_shape: Tuple[int, int]) -> AnimalDetection:
    """Create AnimalDetection from component mask and bbox."""
    min_row, min_col, max_row, max_col = bbox
    image_height, image_width = mask_shape
    
    properties = calculate_mask_properties(component_mask)
    
    rel_bbox = BoundingBox(
        x1=min_col / image_width,
        y1=min_row / image_height,
        x2=(max_col + 1) / image_width,
        y2=(max_row + 1) / image_height
    )
    
    area_ratio = properties['area_ratio']
    bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    compactness = properties['area'] / bbox_area if bbox_area > 0 else 0
    confidence = min(1.0, area_ratio * 10 + compactness * 0.5)
    
    return AnimalDetection(
        bbox=rel_bbox,
        mask=component_mask,
        confidence=confidence,
        area_ratio=area_ratio,
        centroid=properties['centroid']
    )


def extract_detections(mask: np.ndarray) -> List[AnimalDetection]:
    """Extract animal detections from connected components in binary mask.
    
    Args:
        mask: Binary mask with animal regions
        
    Returns:
        List of AnimalDetection objects sorted by confidence
    """
    labeled_mask, num_regions = label(mask)
    detections = []
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_mask == region_id)
        bbox = get_component_bbox(region_mask)
        detection = create_detection_from_component(region_mask, bbox, mask.shape)
        detections.append(detection)
    
    return sorted(detections, key=lambda x: x.confidence, reverse=True)




def inspect_cluster_data(patch_components: np.ndarray, cluster_labels: Dict[str, List[int]], animal_clusters: List[int]) -> None:
    """Inspect cluster data for debugging."""
    print(f"Cluster inspection: shape={patch_components.shape}")
    for cluster_id in animal_clusters:
        if cluster_id < patch_components.shape[1]:
            cluster_data = patch_components[:, cluster_id]
            active_patches = np.sum(cluster_data > 0)
            print(f"  Cluster {cluster_id}: {active_patches} patches, max={cluster_data.max():.3f}")
        else:
            print(f"  ERROR: Cluster {cluster_id} out of bounds")


def detect_animals_from_clusters(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    kernel_size: int = 1
) -> List[AnimalDetection]:
    """
    Detect animals using labeled clusters with morphological cleanup.
    
    Args:
        patch_components: Shape (n_patches, n_components)
        patch_coordinates: Shape (n_patches, 2) with relative coordinates
        cluster_labels: Dictionary mapping semantic labels to cluster indices
        relative_patch_size: Relative size of each patch [0,1]
        background_labels: List of semantic labels to treat as background
        kernel_size: Size of morphological kernel for cleanup
        
    Returns:
        List of AnimalDetection objects
    """
    animal_clusters = collect_animal_clusters(cluster_labels, background_labels)
    if not animal_clusters:
        return []
    
    mask = create_animal_mask(patch_components, patch_coordinates, animal_clusters, relative_patch_size)
    cleaned_mask = apply_morphology(mask, kernel_size)
    return extract_detections(cleaned_mask)


def filter_overlapping_detections(
    detections: List[AnimalDetection],
    iou_threshold: float = 0.3
) -> List[AnimalDetection]:
    """
    Filter overlapping detections using IoU threshold.
    
    Args:
        detections: List of AnimalDetection objects
        iou_threshold: IoU threshold for overlap filtering
        
    Returns:
        Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    filtered = []
    
    for detection in sorted_detections:
        # Check overlap with already accepted detections
        should_keep = True
        
        for accepted in filtered:
            iou = calculate_bbox_iou(detection.bbox, accepted.bbox)
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            filtered.append(detection)
    
    return filtered


def normalize_bbox_to_relative(bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    """Convert absolute pixel coordinates to relative coordinates [0,1].
    
    Args:
        bbox: Bounding box with absolute pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Bounding box with relative coordinates [0,1]
    """
    return BoundingBox(
        x1=bbox.x1 / image_width,
        y1=bbox.y1 / image_height,
        x2=bbox.x2 / image_width,
        y2=bbox.y2 / image_height
    )


def calculate_bbox_iou(bbox1: Tuple[float, float], bbox2: Tuple[float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU value between 0 and 1
    """
    bbox1x1, bbox1y1, bbox1x2, bbox1y2 = bbox1
    bbox2x1, bbox2y1, bbox2x2, bbox2y2 = bbox2
    
    # Calculate intersection
    x1 = max(bbox1x1, bbox2x1)
    y1 = max(bbox1y1, bbox2y1)
    x2 = min(bbox1x2, bbox2x2)
    y2 = min(bbox1y2, bbox2y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (bbox1x2 - bbox1x1) * (bbox1y2 - bbox1y1)
    area2 = (bbox2x2 - bbox2x1) * (bbox2y2 - bbox2y1)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0






def evaluate_animal_detection(
    detections: List[AnimalDetection],
    ground_truth_bbox: BoundingBox,
    image_size: Tuple[int, int] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate animal detection against ground truth bounding box.
    
    Args:
        detections: List of detected animals (relative coordinates)
        ground_truth_bbox: Ground truth bounding box (absolute coordinates)
        image_size: (width, height) - required if ground_truth_bbox is in absolute coordinates
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not detections:
        return {
            'detected': 0,
            'best_iou': 0.0,
            'best_confidence': 0.0,
            'true_positive': 0
        }
    
    # Normalize ground truth bbox if image_size provided
    if image_size is not None:
        width, height = image_size
        # Check if ground truth needs normalization (values > 1 indicate absolute coordinates)
        if ground_truth_bbox.x2 > 1.0 or ground_truth_bbox.y2 > 1.0:
            normalized_gt_bbox = normalize_bbox_to_relative(ground_truth_bbox, width, height)
        else:
            normalized_gt_bbox = ground_truth_bbox
    else:
        normalized_gt_bbox = ground_truth_bbox
    
    # Find best detection by IoU
    best_iou = 0.0
    best_detection = None
    
    for detection in detections:
        iou = calculate_bbox_iou(detection.bbox, normalized_gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_detection = detection
    
    is_true_positive = best_iou >= iou_threshold
    
    return {
        'detected': len(detections),
        'best_iou': best_iou,
        'best_confidence': best_detection.confidence if best_detection else 0.0,
        'true_positive': 1 if is_true_positive else 0
    }