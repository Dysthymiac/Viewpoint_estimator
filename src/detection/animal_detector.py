"""Animal detection using labeled clusters and morphological processing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

from src.models.dinov2_extractor import aggregate_depth_to_patches

from ..data.cvat_loader import BoundingBox
from ..utils.morphological_utils import (
    apply_morphology as morphology_cleanup,
    calculate_mask_properties,
    create_continuous_regions,
    create_spatial_mask_from_patches
)
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import label
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_square_connectivity():
    return [[1,1,1],
            [1,1,1],
            [1,1,1]]

def create_patch_grid(patch_coordinates: np.ndarray, active_patches: np.ndarray) -> np.ndarray:
    """
    Convert patch coordinates and values to 2D grid for morphological operations.
    
    Args:
        patch_coordinates: Shape (n_patches, 2) with relative coordinates
        active_patches: Shape (n_patches,) boolean mask of active patches
        
    Returns:
        2D boolean grid with patches marked
    """
    active_coords = patch_coordinates[active_patches]
    
    if len(active_coords) == 0:
        return np.array([[]], dtype=bool)
    
    # Map coordinates to grid indices
    unique_rows = np.unique(patch_coordinates[:, 0])
    unique_cols = np.unique(patch_coordinates[:, 1])
    
    # Create grid
    grid = np.zeros((len(unique_rows), len(unique_cols)), dtype=bool)
    
    # Fill grid
    for coord in active_coords:
        row_idx = np.where(unique_rows == coord[0])[0][0]
        col_idx = np.where(unique_cols == coord[1])[0][0]
        grid[row_idx, col_idx] = True
    
    return grid




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


def combine_cluster_patches(patch_components: np.ndarray, animal_clusters: List[int]) -> np.ndarray:
    """Combine patches from multiple cluster indices into single mask.
    
    Args:
        patch_components: Shape (n_patches, n_components) with cluster assignments
        animal_clusters: List of cluster indices to include
        
    Returns:
        Combined mask for all specified clusters
    """
    combined_mask = np.zeros(len(patch_components))
    for cluster_id in animal_clusters:
        combined_mask += patch_components[:, cluster_id]  # Let it fail if cluster_id invalid
    return combined_mask


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
    # Combine animal clusters
    combined_mask = combine_cluster_patches(patch_components, animal_clusters)
    
    # Create binary patch values
    patch_values = (combined_mask > 0.1).astype(float)
    
    # Convert to spatial mask using existing function
    return create_spatial_mask_from_patches(patch_coordinates, patch_values, relative_patch_size)


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
    labeled_mask, num_regions = label(mask, structure=get_square_connectivity())
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
    
    combined_mask = combine_cluster_patches(patch_components, animal_clusters)
    patch_values = (combined_mask > 0.1).astype(float)
    mask = create_spatial_mask_from_patches(patch_coordinates, patch_values, relative_patch_size)
    cleaned_mask = morphology_cleanup(mask, kernel_size=kernel_size)
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






def normalize_ground_truth_bbox(ground_truth_bbox: BoundingBox, image_size: Optional[Tuple[int, int]]) -> BoundingBox:
    """Normalize ground truth bbox to relative coordinates if needed."""
    if image_size is None:
        return ground_truth_bbox
    
    width, height = image_size
    # Check if ground truth needs normalization (values > 1 indicate absolute coordinates)
    if ground_truth_bbox.x2 > 1.0 or ground_truth_bbox.y2 > 1.0:
        return normalize_bbox_to_relative(ground_truth_bbox, width, height)
    return ground_truth_bbox


def find_best_detection_by_iou(detections: List[AnimalDetection], normalized_gt_bbox: BoundingBox) -> Tuple[float, Optional[AnimalDetection]]:
    """Find detection with highest IoU against ground truth."""
    best_iou = 0.0
    best_detection = None
    
    for detection in detections:
        iou = calculate_bbox_iou(detection.bbox, normalized_gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_detection = detection
    
    return best_iou, best_detection


def evaluate_animal_detection(
    detections: List[AnimalDetection],
    ground_truth_bbox: BoundingBox,
    image_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate animal detection against ground truth bounding box."""
    if not detections:
        return {
            'detected': 0,
            'best_iou': 0.0,
            'best_confidence': 0.0,
            'true_positive': 0
        }
    
    normalized_gt_bbox = normalize_ground_truth_bbox(ground_truth_bbox, image_size)
    best_iou, best_detection = find_best_detection_by_iou(detections, normalized_gt_bbox)
    is_true_positive = best_iou >= iou_threshold
    
    return {
        'detected': len(detections),
        'best_iou': best_iou,
        'best_confidence': best_detection.confidence if best_detection else 0.0,
        'true_positive': 1 if is_true_positive else 0
    }


def estimate_animal_count_from_body_parts(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    counting_labels: List[str],
    kernel_size: int = 1,
    plot_debug: bool = False
) -> int:
    """
    Estimate number of animals from max count of connected components for each body part.
    """
    counts = []
    
    for clabel in counting_labels:
        if clabel in cluster_labels:
            # Get clusters for this body part
            label_clusters = cluster_labels[clabel]
            
            # Find patches belonging to this body part
            body_part_mask = combine_cluster_patches(patch_components, label_clusters)
            
            # Get active patches
            active_patches = body_part_mask > 0.1
            
            if not np.any(active_patches):
                continue
            
            # Create patch grid and apply morphology
            cleaned_grid = apply_morphological_cleaning_to_patch_grid(
                patch_coordinates, active_patches, kernel_size, pad=True
            )
            
            # Count connected components
            labeled_mask, num_components = label(cleaned_grid, structure=get_square_connectivity())
            counts.append(num_components)
            
            if plot_debug:
                print(f"Debug {clabel}: grid shape {cleaned_grid.shape}, sum {cleaned_grid.sum()}, components {num_components}")
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(cleaned_grid, cmap='gray')
                plt.title(f'{clabel} Mask')
                plt.subplot(1, 2, 2)
                plt.imshow(labeled_mask, cmap='tab10')
                plt.title(f'{clabel} Connected Components ({num_components})')
                plt.tight_layout()
                plt.show()
    
    if not counts:
        return 1  # Default to 1 if no body parts found
    
    return int(np.max(counts))


def prepare_animal_patches(
    patch_components: np.ndarray, 
    cluster_labels: Dict[str, List[int]], 
    background_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract animal patch indices and combined mask."""
    animal_clusters = collect_animal_clusters(cluster_labels, background_labels)
    if not animal_clusters:
        return np.array([]), np.array([])
    
    combined_mask = combine_cluster_patches(patch_components, animal_clusters)
    animal_patch_mask = combined_mask > 0.1
    animal_indices = np.where(animal_patch_mask)[0]
    
    return animal_indices, combined_mask


def create_depth_features(
    patch_coordinates: np.ndarray, 
    depth_values: np.ndarray, 
    animal_indices: np.ndarray,
    depth_mult: float
) -> np.ndarray:
    """Create [x, y, depth] features for k-means clustering."""
    scaled_depths = depth_mult * depth_values
    animal_coords = patch_coordinates[animal_indices]
    animal_depths = scaled_depths[animal_indices]
    
    return np.column_stack([
        animal_coords[:, 1],  # x (col)
        animal_coords[:, 0],  # y (row)
        animal_depths         # depth
    ])


def cluster_animal_patches(features: np.ndarray, n_animals: int) -> np.ndarray:
    """Apply k-means clustering to animal patch features."""
    if len(features) < n_animals:
        n_animals = len(features)  # Can't have more clusters than points
    
    kmeans = KMeans(n_clusters=n_animals)
    return kmeans.fit_predict(features)


def process_animal_cluster(
    cluster_patch_indices: np.ndarray,
    combined_mask: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    kernel_size: int = 1,
    plot_debug: bool = False
) -> List[AnimalDetection]:
    """Process single animal cluster into detections."""
    if len(cluster_patch_indices) == 0:
        return []
    
    # Create binary mask for this cluster
    cluster_mask_1d = np.zeros(len(combined_mask))
    cluster_mask_1d[cluster_patch_indices] = combined_mask[cluster_patch_indices]
    
    # Create patch grid, apply morphology, then convert to spatial mask
    cluster_active_patches = cluster_mask_1d > 0.1
    cleaned_grid = apply_morphological_cleaning_to_patch_grid(
        patch_coordinates, cluster_active_patches, kernel_size, pad=True
    )
    
    # Convert cleaned patch grid back to patch values
    final_patch_values = convert_grid_to_patch_values(cleaned_grid, patch_coordinates)
    
    cleaned_mask = create_spatial_mask_from_patches(patch_coordinates, final_patch_values, relative_patch_size)
    
    if plot_debug:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cleaned_grid, cmap='gray')
        plt.title('Cleaned Mask')
        plt.subplot(1, 3, 2)
        plt.imshow(cleaned_mask, cmap='gray')
        plt.title('After Morphology')
        plt.tight_layout()
        plt.show()
    
    # Extract detections
    return extract_detections(cleaned_mask)


def extract_detections_from_patch_grid(
    cleaned_grid: np.ndarray,
    patch_coordinates: np.ndarray,
    active_patches: np.ndarray,
    relative_patch_size: float
) -> List[AnimalDetection]:
    """Extract single detection from entire cleaned patch grid region."""
    if cleaned_grid.size == 0 or not np.any(cleaned_grid):
        return []
    
    # Treat entire cleaned region as single detection
    active_coords = patch_coordinates[active_patches]
    unique_rows = np.unique(active_coords[:, 0])
    unique_cols = np.unique(active_coords[:, 1])
    
    # Get all coordinates where cleaned grid is active
    region_rows, region_cols = np.where(cleaned_grid)
    
    if len(region_rows) == 0:
        return []
        
    # Map grid indices back to actual coordinates
    actual_coords = []
    for row_idx, col_idx in zip(region_rows, region_cols):
        if row_idx < len(unique_rows) and col_idx < len(unique_cols):
            coord = (unique_rows[row_idx], unique_cols[col_idx])
            actual_coords.append(coord)
    
    if not actual_coords:
        return []
        
    actual_coords = np.array(actual_coords)
    
    # Calculate bounding box from patch coordinates + patch size
    min_row, min_col = actual_coords.min(axis=0)
    max_row, max_col = actual_coords.max(axis=0)
    
    bbox = BoundingBox(
        x1=min_col, y1=min_row,
        x2=max_col + relative_patch_size, y2=max_row + relative_patch_size
    )
    
    # Calculate centroid
    centroid = actual_coords.mean(axis=0)
    
    # Simple confidence based on number of patches
    confidence = min(1.0, len(actual_coords) / 10.0)  # Normalize by expected patch count
    area_ratio = len(actual_coords) / len(active_coords) if len(active_coords) > 0 else 0
    
    # Create dummy mask (empty since we're patch-based now)
    dummy_mask = np.zeros((1, 1), dtype=bool)
    
    detection = AnimalDetection(
        bbox=bbox,
        mask=dummy_mask,
        confidence=confidence,
        area_ratio=area_ratio,
        centroid=(centroid[0], centroid[1])
    )
    
    return [detection]


def detect_animal_regions(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray, 
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    kernel_size: int = 1,
    plot_debug: bool = False
) -> List[Tuple[np.ndarray, BoundingBox]]:
    """Find connected components of animal parts and return crop regions."""
    # Prepare animal patches
    animal_indices, combined_mask = prepare_animal_patches(patch_components, cluster_labels, background_labels)
    if len(animal_indices) == 0:
        return []
    
    # Create patch grid and apply morphology
    animal_patches = combined_mask > 0.1
    cleaned_grid = apply_morphological_cleaning_to_patch_grid(
        patch_coordinates, animal_patches, kernel_size, pad=True
    )
    
    # Find connected components
    labeled_mask, num_regions = label(cleaned_grid, structure=get_square_connectivity())
    
    if plot_debug:
        original_grid = create_patch_grid(patch_coordinates, animal_patches)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(original_grid, cmap='gray')
        plt.title('Animal Patches')
        plt.subplot(1, 3, 2)
        plt.imshow(cleaned_grid, cmap='gray')
        plt.title('After Morphology')
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        plt.imshow(labeled_mask, cmap='tab10')
        plt.title(f'Connected Components ({num_regions})')
        plt.colorbar()
        plt.show()
    
    # Convert each component back to bounding box in original coordinates
    regions = []
    unique_rows = np.unique(patch_coordinates[:, 0])
    unique_cols = np.unique(patch_coordinates[:, 1])
    
    for region_id in range(1, num_regions + 1):
        region_mask = (labeled_mask == region_id)
        region_rows, region_cols = np.where(region_mask)
        
        if len(region_rows) == 0:
            continue
        
        # Convert grid coordinates back to patch coordinates
        min_row_coord = unique_rows[region_rows.min()]
        max_row_coord = unique_rows[region_rows.max()]
        min_col_coord = unique_cols[region_cols.min()]
        max_col_coord = unique_cols[region_cols.max()]
        
        # Create bounding box with padding based on patch size and kernel
        padding = kernel_size/2 * relative_patch_size
        bbox = BoundingBox(
            x1=max(0, min_col_coord - padding),
            y1=max(0, min_row_coord - padding), 
            x2=min(1, max_col_coord + relative_patch_size + padding),
            y2=min(1, max_row_coord + relative_patch_size + padding)
        )
        
        regions.append((region_mask, bbox))
    
    return regions



def extract_patches(image, extractor, analysis_method, n_clusters):
    patch_features, patch_coordinates, actual_relative_patch_size, depthmap = extractor.extract_patch_features(image)
    patch_depths = aggregate_depth_to_patches(depthmap, patch_coordinates, actual_relative_patch_size, 'min')
            
    # Transform with analysis method
    raw_output = analysis_method.transform(patch_features)
    if raw_output.ndim == 1:
        patch_components = np.zeros((len(raw_output), n_clusters))
        patch_components[np.arange(len(raw_output)), raw_output] = 1.0
    else:
        patch_components = raw_output
    return patch_components, patch_coordinates, actual_relative_patch_size, patch_depths, depthmap


def crop_image_region(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Crop region from image using relative coordinates."""
    w, h = image.size
    x1 = int(bbox.x1 * w)
    y1 = int(bbox.y1 * h) 
    x2 = int(bbox.x2 * w)
    y2 = int(bbox.y2 * h)
    
    return image.crop((x1, y1, x2, y2))


def hierarchical_animal_detection(
    image: np.ndarray,
    extractor,
    analysis_method, 
    n_clusters: int,
    cluster_labels: Dict[str, List[int]],
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    counting_labels: List[str] = ["body", "thigh", "neck", "belly", "back", "head"],
    kernel_size: int = 1,
    depth_mult: float = 1,
    plot_debug: bool = False
) -> List[AnimalDetection]:
    """Hierarchical animal detection with multi-scale processing."""
    # Step 1: Extract patches from full image
    patch_components, patch_coordinates, relative_patch_size, patch_depths, depthmap = extract_patches(
        image, extractor, analysis_method, n_clusters
    )
    
    # Step 2: Find connected components of animal parts
    regions = detect_animal_regions(
        patch_components, patch_coordinates, cluster_labels, relative_patch_size, background_labels, kernel_size, plot_debug
    )
    
    if len(regions) == 0:
        return []
    
    # Step 3 & 4: Crop each region and apply detection
    all_detections = []
    
    for i, (region_mask, bbox) in enumerate(regions):
        
        # Crop image region
        cropped_image = crop_image_region(image, bbox)
        if plot_debug:
            print(f"Processing region {i+1}/{len(regions)}: {bbox}")
            plt.figure()
            plt.imshow(cropped_image)
            plt.show()
        
        if cropped_image.size == 0:
            continue
            
        # Extract patches from cropped region
        crop_components, crop_coordinates, crop_patch_size, crop_depths, _ = extract_patches(
            cropped_image, extractor, analysis_method, n_clusters
        )
        
        # Apply depth-based detection to cropped region
        crop_detections = detect_animals_with_depth(
            crop_components, crop_coordinates, crop_depths, cluster_labels, 
            crop_patch_size, background_labels, counting_labels, kernel_size, depth_mult, plot_debug
        )
        
        # Adjust detection coordinates back to full image space
        for detection in crop_detections:
            detection.bbox = BoundingBox(
                x1=bbox.x1 + detection.bbox.x1 * (bbox.x2 - bbox.x1),
                y1=bbox.y1 + detection.bbox.y1 * (bbox.y2 - bbox.y1),
                x2=bbox.x1 + detection.bbox.x2 * (bbox.x2 - bbox.x1),
                y2=bbox.y1 + detection.bbox.y2 * (bbox.y2 - bbox.y1)
            )
        
        all_detections.extend(crop_detections)
    
    return sorted(all_detections, key=lambda x: x.confidence, reverse=True)


def detect_animals_with_depth(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    depth_values: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    counting_labels: List[str] = ["body", "thigh", "neck", "belly", "back", "head"],
    kernel_size: int = 1,
    depth_mult: float = 100,
    plot_debug: bool = False
) -> List[AnimalDetection]:
    """Detect animals using depth-based k-means clustering."""
    # Prepare animal patches
    animal_indices, combined_mask = prepare_animal_patches(patch_components, cluster_labels, background_labels)
    if len(animal_indices) == 0:
        return []
    
    # Create depth features for clustering
    features = create_depth_features(patch_coordinates, depth_values, animal_indices, depth_mult)
    
    # Estimate number of animals and apply clustering
    n_animals = estimate_animal_count_from_body_parts(patch_components, patch_coordinates, cluster_labels, counting_labels, kernel_size, plot_debug)
    cluster_assignments = cluster_animal_patches(features, n_animals)
    
    if plot_debug:
        # Create single image showing all cluster assignments
        all_animal_patches = combined_mask > 0.1
        cluster_assignment_grid = create_patch_grid(patch_coordinates, all_animal_patches)
        
        # Fill with cluster IDs
        for i in range(n_animals):
            cluster_mask = cluster_assignments == i
            cluster_patch_indices = animal_indices[cluster_mask]
            
            # Create mask for this cluster
            cluster_mask_1d = np.zeros(len(combined_mask))
            cluster_mask_1d[cluster_patch_indices] = i + 1
            cluster_active_patches = cluster_mask_1d > 0.5
            cluster_grid = create_patch_grid(patch_coordinates, cluster_active_patches)
            cluster_assignment_grid = cluster_assignment_grid + cluster_grid * (i+1)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cluster_assignment_grid, cmap='tab10')
        plt.title('Cluster Assignments')
        plt.colorbar()
        plt.show()
    
    # Process each cluster
    detections = []
    for cluster_id in range(n_animals):
        cluster_mask = cluster_assignments == cluster_id
        cluster_patch_indices = animal_indices[cluster_mask]
        
        cluster_detections = process_animal_cluster(
            cluster_patch_indices, combined_mask, patch_coordinates, relative_patch_size, kernel_size, plot_debug
        )
        detections.extend(cluster_detections)
    
    return sorted(detections, key=lambda x: x.confidence, reverse=True)


# ===== SAM2 Integration Functions =====

def apply_morphological_cleaning_to_patch_grid(
    patch_coordinates: np.ndarray,
    active_patches: np.ndarray,
    kernel_size: int,
    pad: bool = True
) -> np.ndarray:
    """Apply morphological cleaning to a patch grid."""
    if kernel_size <= 0:
        return create_patch_grid(patch_coordinates, active_patches)
    
    patch_grid = create_patch_grid(patch_coordinates, active_patches)
    return morphology_cleanup(patch_grid, kernel_size=kernel_size, pad=pad)


def convert_grid_to_patch_indices(
    cleaned_grid: np.ndarray,
    patch_coordinates: np.ndarray
) -> List[int]:
    """Convert a cleaned patch grid back to patch indices."""
    unique_rows = np.unique(patch_coordinates[:, 0])
    unique_cols = np.unique(patch_coordinates[:, 1])
    
    patch_indices = []
    grid_rows, grid_cols = np.where(cleaned_grid)
    
    for row_idx, col_idx in zip(grid_rows, grid_cols):
        if row_idx < len(unique_rows) and col_idx < len(unique_cols):
            coord = (unique_rows[row_idx], unique_cols[col_idx])
            # Find the patch index that matches this coordinate
            for i, patch_coord in enumerate(patch_coordinates):
                if np.allclose(patch_coord, coord, atol=1e-6):
                    patch_indices.append(i)
                    break
    
    return patch_indices


def convert_grid_to_patch_values(
    cleaned_grid: np.ndarray,
    patch_coordinates: np.ndarray
) -> np.ndarray:
    """Convert a cleaned patch grid back to patch values."""
    unique_rows = np.unique(patch_coordinates[:, 0])
    unique_cols = np.unique(patch_coordinates[:, 1])
    
    patch_values = np.zeros(len(patch_coordinates))
    for i, coord in enumerate(patch_coordinates):
        row_idx = np.where(unique_rows == coord[0])[0][0]
        col_idx = np.where(unique_cols == coord[1])[0][0]
        if row_idx < cleaned_grid.shape[0] and col_idx < cleaned_grid.shape[1]:
            patch_values[i] = cleaned_grid[row_idx, col_idx]
    
    return patch_values


def apply_morphological_cleaning_to_patches(
    patch_indices: List[int],
    patch_coordinates: np.ndarray,
    kernel_size: int
) -> List[int]:
    """Apply morphological cleaning to patch indices and return cleaned indices."""
    if kernel_size <= 0 or len(patch_indices) == 0:
        return patch_indices
    
    # Create patch mask
    patch_mask = np.zeros(len(patch_coordinates), dtype=bool)
    patch_mask[patch_indices] = True
    
    # Apply morphological cleaning
    cleaned_grid = apply_morphological_cleaning_to_patch_grid(
        patch_coordinates, patch_mask, kernel_size
    )
    
    # Convert back to patch indices
    cleaned_indices = convert_grid_to_patch_indices(cleaned_grid, patch_coordinates)
    
    return cleaned_indices if len(cleaned_indices) > 0 else patch_indices


def calculate_body_part_centroids(
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    kernel_size: int = 1,
    body_part_labels: List[str] = ["body", "head", "neck", "back", "belly", "thigh"]
) -> List[Tuple[np.ndarray, str]]:
    """Calculate centroids of body parts with morphological cleanup and connected components."""
    
    centroids = []
    
    for clabel in body_part_labels:
        if clabel in cluster_labels:
            # Get patches for this body part
            label_clusters = cluster_labels[clabel]
            body_part_mask = combine_cluster_patches(patch_components, label_clusters)
            active_patches = body_part_mask > 0.1
            
            if not np.any(active_patches):
                continue
            
            # Step 1: Apply morphological cleanup
            cleaned_grid = apply_morphological_cleaning_to_patch_grid(
                patch_coordinates, active_patches, kernel_size
            )
            
            # Step 2: Find connected components
            labeled_mask, num_components = label(cleaned_grid)
            
            # Step 3: Calculate centroid for each connected component
            unique_rows = np.unique(patch_coordinates[:, 0])
            unique_cols = np.unique(patch_coordinates[:, 1])
            
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_mask == component_id)
                component_rows, component_cols = np.where(component_mask)
                
                if len(component_rows) == 0:
                    continue
                
                # Convert grid coordinates back to patch coordinates for this component
                component_patch_coords = []
                for row_idx, col_idx in zip(component_rows, component_cols):
                    if row_idx < len(unique_rows) and col_idx < len(unique_cols):
                        coord = (unique_rows[row_idx], unique_cols[col_idx])
                        component_patch_coords.append(coord)
                
                if len(component_patch_coords) == 0:
                    continue
                    
                component_patch_coords = np.array(component_patch_coords) + relative_patch_size*0.5
                
                # Calculate centroid of this connected component
                centroid_coord = component_patch_coords.mean(axis=0)
                
                # Find the actual patch closest to centroid (to avoid holes)
                distances = ((component_patch_coords - centroid_coord) ** 2).sum(axis=1)
                closest_patch_idx = np.argmin(distances)
                actual_centroid = component_patch_coords[closest_patch_idx] #+ relative_patch_size*0.5
                
                centroids.append((actual_centroid, clabel))
    
    return centroids


def filter_background_masks(
    sam2_results: List[Dict], 
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    image_shape: Tuple[int, int],
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    background_threshold: float = 0.6
) -> List[Dict]:
    """Filter out masks that contain mostly background patches."""
    if len(sam2_results) == 0:
        return sam2_results
    
    # Get background patch mask
    background_clusters = []
    for label in background_labels:
        if label in cluster_labels:
            background_clusters.extend(cluster_labels[label])
    
    if len(background_clusters) == 0:
        return sam2_results  # No background clusters defined
    
    background_mask = combine_cluster_patches(patch_components, background_clusters)
    background_patch_indices = np.where(background_mask > 0.1)[0]
    
    h, w = image_shape
    filtered_results = []
    
    for result in sam2_results:
        mask = result['mask']
        
        # Find patches that overlap with this SAM2 mask
        overlapping_patch_indices = []
        
        for i, coord in enumerate(patch_coordinates):
            # Convert relative coordinates to pixel coordinates
            y_rel, x_rel = coord
            y_px = int(y_rel * h)
            x_px = int(x_rel * w)
            
            # Calculate patch bounds in pixels
            patch_h = max(1, int(relative_patch_size * max(h, w)))
            patch_w = max(1, int(relative_patch_size * max(h, w)))
            
            y1 = max(0, y_px)
            y2 = min(h, y_px + patch_h)
            x1 = max(0, x_px)
            x2 = min(w, x_px + patch_w)
            
            # Extract patch region from SAM2 mask
            patch_mask = mask[y1:y2, x1:x2]
            
            if patch_mask.size == 0:
                continue
            
            # Check if patch has significant overlap with mask
            overlap_ratio = np.sum(patch_mask) / patch_mask.size
            
            if overlap_ratio > 0.5:  # At least 50% of patch is covered
                overlapping_patch_indices.append(i)
        
        if len(overlapping_patch_indices) == 0:
            continue
        
        # Calculate what fraction of overlapping patches are background
        background_count = sum(1 for idx in overlapping_patch_indices if idx in background_patch_indices)
        background_ratio = background_count / len(overlapping_patch_indices)
        
        # Keep mask only if background ratio is below threshold
        if background_ratio < background_threshold:
            filtered_results.append(result)
    
    return filtered_results


def merge_overlapping_masks(sam2_results: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
    """Merge overlapping SAM2 masks based on overlap ratio to smallest mask."""
    if len(sam2_results) <= 1:
        return sam2_results
    
    merged_results = []
    used_indices = set()
    
    for i, result1 in enumerate(sam2_results):
        if i in used_indices:
            continue
            
        mask1 = result1['mask']
        mask1_area = np.sum(mask1)
        
        # Find overlapping masks
        overlapping_indices = [i]
        merged_mask = mask1.copy()
        merged_score = result1['score']
        merged_body_parts = [result1['body_part']]
        
        for j, result2 in enumerate(sam2_results[i+1:], start=i+1):
            if j in used_indices:
                continue
                
            mask2 = result2['mask']
            mask2_area = np.sum(mask2)
            
            # Calculate overlap
            intersection = np.sum(mask1 & mask2)
            smaller_area = min(mask1_area, mask2_area)
            
            if smaller_area > 0:
                overlap_ratio = intersection / smaller_area
                
                if overlap_ratio > overlap_threshold:
                    # Merge masks
                    overlapping_indices.append(j)
                    merged_mask = merged_mask | mask2
                    merged_score = max(merged_score, result2['score'])
                    if result2['body_part'] not in merged_body_parts:
                        merged_body_parts.append(result2['body_part'])
        
        # Mark indices as used
        used_indices.update(overlapping_indices)
        
        # Create merged result
        merged_result = {
            'mask': merged_mask,
            'score': merged_score,
            'body_part': '+'.join(merged_body_parts) if len(merged_body_parts) > 1 else merged_body_parts[0],
            'merged_from': overlapping_indices,
            'point_index': result1['point_index']
        }
        
        # Keep centroid_pixel from the highest scoring mask
        best_idx = overlapping_indices[0]
        best_score = sam2_results[best_idx]['score']
        for idx in overlapping_indices[1:]:
            if sam2_results[idx]['score'] > best_score:
                best_score = sam2_results[idx]['score']
                best_idx = idx
        merged_result['centroid_pixel'] = sam2_results[best_idx]['centroid_pixel']
        
        merged_results.append(merged_result)
    
    return merged_results


def apply_sam2_with_body_part_points(
    image: np.ndarray,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    sam2_predictor: SAM2ImagePredictor,
    kernel_size: int = 2,
    body_part_labels: List[str] = ["body", "head", "neck", "back", "belly", "thigh"],
    score_threshold: float = 0.8,
    merge_overlap_threshold: float = 0.5,
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    background_threshold: float = 0.6
) -> List[Dict]:
    """Apply SAM2 with batched body part centroids as point prompts."""
    
    # Set image in SAM2 predictor
    sam2_predictor.set_image(image)
    
    # Step 1: Calculate body part centroids
    body_part_centroids = calculate_body_part_centroids(
        patch_components, 
        patch_coordinates, 
        cluster_labels, 
        relative_patch_size,
        kernel_size=kernel_size, 
        body_part_labels=body_part_labels
    )
    
    if len(body_part_centroids) == 0:
        return []
    
    # Step 2: Convert centroids to pixel coordinates and create batched points (b x 1 x 2)
    h, w = image.shape[:2]
    batched_points = []
    centroid_labels = []
    # plt.figure()
    # plt.imshow(image)
    for centroid_coord, part_label in body_part_centroids:
        # Convert relative coordinates to pixels
        cent_x = int(centroid_coord[1] * w)  # x from column coord
        cent_y = int(centroid_coord[0] * h)  # y from row coord
        # plt.scatter(cent_x, cent_y, c="r")
        # plt.text(cent_x, cent_y, part_label)
        # Each point gets its own "batch" - shape will be (b, 1, 2)
        batched_points.append([[cent_x, cent_y]])
        centroid_labels.append(part_label)
    # plt.show()
    # Convert to numpy array - shape: (b, 1, 2) 
    input_points = np.array(batched_points)
    input_labels = np.ones((len(batched_points), 1))  # All positive points, shape: (b, 1)
    
    # Step 3: Single SAM2 call with batched points
    masks, scores, logits = sam2_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False  # Single mask per point
    )
    
    # Step 4: Process results
    sam2_results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score > score_threshold:
            sam2_results.append({
                'mask': mask.squeeze() > 0,
                'score': score.squeeze(),
                'centroid_pixel': batched_points[i][0],  # [x, y] in pixels
                'body_part': centroid_labels[i],
                'point_index': i
            })
    
    # Step 5: Merge overlapping masks
    merged_results = merge_overlapping_masks(sam2_results, merge_overlap_threshold)
    
    # Step 6: Filter out masks with mostly background patches
    filtered_results = filter_background_masks(
        merged_results, patch_components, patch_coordinates, cluster_labels,
        relative_patch_size, image.shape[:2], background_labels, background_threshold
    )
    
    return filtered_results


def map_sam2_masks_to_patches(
    sam2_results: List[Dict],
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    image_shape: Tuple[int, int],
    kernel_size: int = 1
) -> List[Dict]:
    """Map SAM2 masks to patch indices."""
    
    h, w = image_shape
    detections = []
    
    for result in sam2_results:
        mask = result['mask']
        score = result['score']
        
        # Find patches that overlap with this mask
        patch_indices = []
        
        for i, coord in enumerate(patch_coordinates):
            # Convert relative coordinates to pixel coordinates
            y_rel, x_rel = coord
            y_px = int(y_rel * h)
            x_px = int(x_rel * w)
            
            # Calculate patch bounds in pixels
            patch_h = max(1, int(relative_patch_size * max(w, h)))
            patch_w = max(1, int(relative_patch_size * max(w, h)))
            
            y1 = max(0, y_px)
            y2 = min(h, y_px + patch_h)
            x1 = max(0, x_px)
            x2 = min(w, x_px + patch_w)
            
            # Extract patch region from SAM2 mask
            patch_mask = mask[y1:y2, x1:x2]
            
            if patch_mask.size == 0:
                continue
            
            # Check if patch has significant overlap with mask
            overlap_ratio = np.sum(patch_mask) / patch_mask.size
            
            if overlap_ratio > 0.5:  # At least 50% of patch is covered
                patch_indices.append(i)
        
        if len(patch_indices) > 0:
            # Apply morphological cleaning to patch indices
            patch_indices = apply_morphological_cleaning_to_patches(
                patch_indices, patch_coordinates, kernel_size
            )
            
            if len(patch_indices) == 0:
                continue
                
            # Calculate centroid from patch coordinates
            patch_coords = patch_coordinates[patch_indices]
            centroid = patch_coords.mean(axis=0)
            
            # Create bounding box from mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0:
                bbox = BoundingBox(
                    x1=x_indices.min() / w,
                    y1=y_indices.min() / h,
                    x2=x_indices.max() / w,
                    y2=y_indices.max() / h
                )
            else:
                # Fallback: create bbox from patch coordinates
                bbox = BoundingBox(
                    x1=patch_coords[:, 1].min(),
                    y1=patch_coords[:, 0].min(),
                    x2=patch_coords[:, 1].max(),
                    y2=patch_coords[:, 0].max()
                )
            
            detection = AnimalDetection(
                bbox=bbox,
                mask=mask,
                confidence=score,
                area_ratio=len(patch_indices) / len(patch_coordinates),
                centroid=(centroid[0], centroid[1])
            )
            
            detections.append({
                'detection': detection,
                'patch_indices': np.array(patch_indices),
                'sam2_score': score,
                'body_part': result.get('body_part', 'unknown')
            })
    
    return detections


def detect_animals_with_sam2(
    image: Image.Image,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    cluster_labels: Dict[str, List[int]],
    relative_patch_size: float,
    sam2_predictor: SAM2ImagePredictor,
    score_threshold: float = 0.8,
    merge_overlap_threshold: float = 0.5,
    kernel_size: int = 1,
    body_part_labels: List[str] = ["body", "head", "neck", "back", "belly", "thighs"],
    background_labels: List[str] = ["background", "vegetation", "ground", "shadow"],
    background_threshold: float = 0.6,
    image_resize: Optional[int] = None,
    plot_debug: bool = False
) -> List[Dict]:
    """New SAM2-based animal detection pipeline."""
    if image_resize is not None:
        original_width, original_height = image.size
        if original_width >= original_height:
            new_width = image_resize
            new_height = int(original_height * image_resize / original_width)
        else: 
            new_height = image_resize
            new_width = int(original_width * image_resize / original_height)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
    image = np.array(image.convert("RGB"))
    
    # Step 1: Apply SAM2 with body part centroids as point prompts
    sam2_results = apply_sam2_with_body_part_points(
        image, patch_components, patch_coordinates, cluster_labels, relative_patch_size,
        sam2_predictor, score_threshold=score_threshold, 
        merge_overlap_threshold=merge_overlap_threshold,
        kernel_size=kernel_size, body_part_labels=body_part_labels,
        background_labels=background_labels, background_threshold=background_threshold
    )
    
    if plot_debug and len(sam2_results) > 0:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        # Show all SAM2 masks
        plt.subplot(1, 3, 2)
        combined_mask = np.zeros(image.shape[:2])
        for result in sam2_results:
            combined_mask += result['mask']
        plt.imshow(image)
        plt.imshow(combined_mask, alpha=0.5, cmap='Reds')
        plt.title(f'SAM2 Masks (score > {score_threshold})')
        plt.axis('off')
        
        # Show individual masks with scores
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sam2_results)))
        for i, result in enumerate(sam2_results):
            mask_colored = np.zeros((*result['mask'].shape, 4))
            mask_colored[result['mask'] > 0] = [*colors[i][:3], 0.6]
            plt.gca().imshow(mask_colored)
            plt.scatter(*result['centroid_pixel'], c='red')
            # Add score text
            y, x = np.where(result['mask'])
            if len(y) > 0:
                plt.text(x.mean(), y.mean(), f'{result["score"]:.2f}', 
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('Individual SAM2 Masks with Scores')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Step 2: Map SAM2 masks to patch indices
    detections = map_sam2_masks_to_patches(
        sam2_results, patch_coordinates, relative_patch_size, image.shape[:2], kernel_size
    )
    
    if plot_debug and len(detections) > 0:
        plt.figure(figsize=(12, 8))
        for i, det_info in enumerate(detections):
            plt.subplot(2, 3, i + 1)
            plt.imshow(image)
            
            # Show patches for this detection
            patch_indices = det_info['patch_indices']
            patch_coords = patch_coordinates[patch_indices]
            h, w = image.shape[:2]
            
            for coord in patch_coords:
                y_rel, x_rel = coord
                y_px = int(y_rel * h)
                x_px = int(x_rel * w)
                patch_h = max(1, int(relative_patch_size * max(h, w)))
                patch_w = max(1, int(relative_patch_size * max(h, w)))
                
                plt.gca().add_patch(plt.Rectangle((x_px, y_px), patch_w, patch_h,
                                                fill=False, color='yellow', linewidth=1))
            
            # Overlay SAM2 mask
            mask = det_info['detection'].mask
            plt.imshow(mask, alpha=0.4, cmap='Reds')
            
            plt.title(f'Detection {i+1}\nScore: {det_info["sam2_score"]:.2f}\nPatches: {len(patch_indices)}')
            plt.axis('off')
            
            if i >= 5:  # Limit to 6 subplots
                break
        
        plt.tight_layout()
        plt.show()
    
    return detections


def get_body_part_bounding_box(
    detection_info: Dict,
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    body_parts: List[str],
    image_size: Tuple[int, int],
    padding: float = 0.0,
    apply_morphology: bool = True,
    kernel_size: int = 1
) -> Optional[BoundingBox]:
    """
    Get bounding box from specific body parts within a detection.
    
    Args:
        detection_info: Detection dictionary from detect_animals_with_sam2()
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        relative_patch_size: float
        cluster_labels: Dict mapping body part names to cluster ID lists
        body_parts: List of body part names to include in bounding box
        image_size: (width, height) tuple
        padding: Additional padding around the bounding box (relative to image size)
        apply_morphology: Whether to apply morphological cleaning to patches
        kernel_size: Size of morphological kernel
        
    Returns:
        BoundingBox object or None if no valid body parts found
    """
    patch_indices = detection_info['patch_indices']
    
    # Filter patch components to only include patches from this detection
    filtered_patch_components = patch_components[patch_indices]
    filtered_patch_coordinates = patch_coordinates[patch_indices]
    
    # Collect patch indices for specified body parts
    body_part_patch_indices = []
    w, h = image_size
    relative_patch_w = (relative_patch_size * max(w, h))/w
    relative_patch_h = (relative_patch_size * max(w, h))/h
    
    for part_name in body_parts:
        if part_name in cluster_labels:
            # Get cluster IDs for this body part
            cluster_ids = cluster_labels[part_name]
            
            # Find patches that belong to these clusters
            for i, coord in enumerate(filtered_patch_coordinates):
                for cluster_id in cluster_ids:
                    if cluster_id < filtered_patch_components.shape[1]:
                        if filtered_patch_components[i, cluster_id] > 0.1:  # Threshold for membership
                            body_part_patch_indices.append(i)
                            break
    
    if len(body_part_patch_indices) == 0:
        return None
    
    # Apply morphological cleaning if requested
    if apply_morphology and kernel_size > 0:
        cleaned_indices = apply_morphological_cleaning_to_patches(
            body_part_patch_indices, filtered_patch_coordinates, kernel_size
        )
        if len(cleaned_indices) == 0:
            return None
        body_part_coordinates = filtered_patch_coordinates[cleaned_indices]
    else:
        body_part_coordinates = filtered_patch_coordinates[body_part_patch_indices]
    
    if len(body_part_coordinates) == 0:
        return None
    
    # body_part_coordinates is already a numpy array
    coords = body_part_coordinates
    
    # Calculate bounding box in relative coordinates
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    
    # Add padding
    width_padding = padding * (max_x - min_x) if max_x > min_x else padding * 0.1
    height_padding = padding * (max_y - min_y) if max_y > min_y else padding * 0.1
    
    x1 = max(0.0, min_x - width_padding)
    y1 = max(0.0, min_y - height_padding)
    x2 = min(1.0, max_x + width_padding + relative_patch_w)
    y2 = min(1.0, max_y + height_padding + relative_patch_h)
    
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def get_detection_body_part_boxes(
    detections: List[Dict],
    patch_components: np.ndarray,
    patch_coordinates: np.ndarray,
    relative_patch_size: float,
    cluster_labels: Dict[str, List[int]],
    image_size: Tuple[int, int],
    body_parts: List[str] = None,
    padding: float = 0.1,
    apply_morphology: bool = True,
    kernel_size: int = 1
) -> List[Dict]:
    """
    Get bounding boxes for different body part groups for all detections.
    
    Args:
        detections: List of detection dictionaries from detect_animals_with_sam2()
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        patch_coordinates: Patch center coordinates (n_patches, 2) in relative space [0,1]
        relative_patch_size: float
        cluster_labels: Dict mapping body part names to cluster ID lists
        image_size: (width, height) tuple
        body_part_groups: Dict mapping group names to list of body parts
                         Default: {"head": ["head", "neck"], "body": ["body", "back", "belly"], "full": ["head", "neck", "body", "back", "belly", "legs"]}
        padding: Additional padding around bounding boxes
        apply_morphology: Whether to apply morphological cleaning to patches
        kernel_size: Size of morphological kernel
        
    Returns:
        List of detection dictionaries enriched with body part bounding boxes
    """
    
    enriched_detections = []
    
    for detection_info in detections:
        enriched_detection = detection_info.copy()
        enriched_detection['body_part_boxes'] = {}
        
        bbox = get_body_part_bounding_box(
            detection_info, patch_components, patch_coordinates, relative_patch_size,
            cluster_labels, body_parts, image_size, padding, apply_morphology, kernel_size
        )
        enriched_detection['body_part_box'] = bbox
        
        enriched_detections.append(enriched_detection)
    
    return enriched_detections
