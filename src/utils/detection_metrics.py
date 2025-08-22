"""Pure calculation functions for detection metrics following style guide."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data.cvat_loader import BoundingBox
from ..detection.animal_detector import AnimalDetection


def calculate_bbox_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate IoU between two bounding boxes."""
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
    area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_detection_accuracy(detections: List[AnimalDetection], ground_truth_bbox: BoundingBox, iou_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate detection accuracy metrics."""
    if not detections:
        return {
            'detected': 0,
            'best_iou': 0.0,
            'best_confidence': 0.0,
            'true_positive': 0
        }
    
    best_iou = 0.0
    best_confidence = 0.0
    
    for detection in detections:
        iou = calculate_bbox_iou(detection.bbox, ground_truth_bbox)
        if iou > best_iou:
            best_iou = iou
            best_confidence = detection.confidence
    
    return {
        'detected': len(detections),
        'best_iou': best_iou,
        'best_confidence': best_confidence,
        'true_positive': 1 if best_iou >= iou_threshold else 0
    }


def calculate_aspect_ratio(width: float, height: float) -> float:
    """Calculate aspect ratio (width/height)."""
    return width / height if height > 0 else 0.0


def calculate_euclidean_distance(point1: tuple, point2: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_centroid(coordinates: np.ndarray) -> tuple:
    """Calculate centroid of coordinate array."""
    if len(coordinates) == 0:
        return (0.0, 0.0)
    return (float(np.mean(coordinates[:, 0])), float(np.mean(coordinates[:, 1])))


def calculate_spread(coordinates: np.ndarray) -> tuple:
    """Calculate x and y spread of coordinates."""
    if len(coordinates) < 2:
        return (0.0, 0.0)
    return (float(np.std(coordinates[:, 1])), float(np.std(coordinates[:, 0])))


def calculate_bounding_box_from_coords(coordinates: np.ndarray) -> tuple:
    """Calculate bounding box from coordinates."""
    if len(coordinates) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    min_y, min_x = coordinates.min(axis=0)
    max_y, max_x = coordinates.max(axis=0)
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def is_linear_arrangement(centroids: List[tuple], threshold: float = 0.8) -> float:
    """Calculate linearity score for centroid arrangement."""
    if len(centroids) < 3:
        return 0.5 if len(centroids) == 2 else 0.0
    
    centroids = np.array(centroids)
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(centroids) - 1):
        distances.append(calculate_euclidean_distance(centroids[i], centroids[i + 1]))
    
    # Check triangle inequality for linearity
    d01, d02, d12 = distances[0], distances[1], distances[2] if len(distances) >= 3 else 0
    max_dist = max(d01, d02, d12)
    
    if max_dist == 0:
        return 0.0
    
    sum_two_smaller = d01 + d02 + d12 - max_dist
    linearity = 1.0 - abs(max_dist - sum_two_smaller) / max_dist
    return max(0.0, linearity)


def calculate_symmetry_score(x_coordinates: List[float]) -> float:
    """Calculate bilateral symmetry score."""
    if len(x_coordinates) < 2:
        return 0.0
    
    x_coords = np.array(x_coordinates)
    mean_x = np.mean(x_coords)
    deviations = np.abs(x_coords - mean_x)
    
    max_deviation = 0.5  # Maximum possible deviation
    symmetry = 1.0 - np.mean(deviations) / max_deviation
    return max(0.0, symmetry)


def calculate_axis_ratio_from_pca(coordinates: np.ndarray) -> tuple:
    """Calculate primary/secondary axis ratio using PCA."""
    if len(coordinates) < 2:
        return (0.0, 1.0)
    
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    pca.fit(coordinates)
    
    explained_var = pca.explained_variance_ratio_
    primary_angle = np.arctan2(pca.components_[0][0], pca.components_[0][1])
    
    axis_ratio = explained_var[0] / explained_var[1] if len(explained_var) > 1 and explained_var[1] > 0 else 10.0
    
    return (float(primary_angle), float(axis_ratio))


def normalize_coordinates(coordinates: np.ndarray, bounds: tuple) -> np.ndarray:
    """Normalize coordinates to [0,1] range."""
    min_y, min_x, max_y, max_x = bounds
    
    if len(coordinates) == 0:
        return coordinates
    
    normalized = coordinates.copy().astype(float)
    
    if max_y > min_y:
        normalized[:, 0] = (coordinates[:, 0] - min_y) / (max_y - min_y)
    else:
        normalized[:, 0] = 0.5
        
    if max_x > min_x:
        normalized[:, 1] = (coordinates[:, 1] - min_x) / (max_x - min_x)
    else:
        normalized[:, 1] = 0.5
    
    return normalized