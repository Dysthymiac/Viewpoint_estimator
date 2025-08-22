"""
Dataset Evaluation Script for Detection and Viewpoint Classification

Processes the entire dataset to:
1. Calculate detection bounding boxes for all images
2. Classify viewpoints using 3D analysis with depth information
3. Compare predictions to ground truth annotations
4. Generate comprehensive statistics and visualizations
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
from tqdm import tqdm

from src.data.cvat_loader import CVATLoader, BoundingBox
from src.models.dinov2_extractor import DINOv2PatchExtractor
from src.utils.analysis_utils import create_analysis_method_from_config, load_analysis_model
from src.utils.config import load_config
from src.detection.animal_detector import calculate_bbox_iou, detect_animals_from_clusters
from src.classification.viewpoint_3d import estimate_viewpoint_with_axis_fitting
from src.utils.detection_utils import filter_patches_for_detection, validate_detection_patches






def evaluate_single_image(annotation, 
                         loader: CVATLoader,
                         extractor: DINOv2PatchExtractor, 
                         analysis_method,
                         cluster_labels: Dict[str, List[int]],
                         config: Dict) -> Dict:
    """
    Evaluate detection and viewpoint classification for a single image.
    
    Args:
        annotation: CVAT annotation object
        loader: CVAT loader to load images
        extractor: DINOv2 feature extractor
        analysis_method: Fitted clustering/analysis method
        cluster_labels: Mapping from body parts to cluster IDs
        config: Configuration dictionary
        
    Returns:
        Dictionary with evaluation results
    """
    # Load image using loader
    image = loader.load_image(annotation)
    
    # Extract features and depth
    patch_features, patch_coordinates, relative_patch_size, depth_map = extractor.extract_patch_features_with_depth(image)
    
    # Convert depth map to per-patch values if depth is available
    patch_depths = None
    if depth_map is not None:
        from src.models.dinov2_extractor import aggregate_depth_to_patches
        patch_depths = aggregate_depth_to_patches(depth_map, patch_coordinates, relative_patch_size)
    
    # Get cluster assignments
    if hasattr(analysis_method, 'predict'):
        cluster_assignments = analysis_method.predict(patch_features)
    elif hasattr(analysis_method, 'transform'):
        cluster_assignments = analysis_method.transform(patch_features)
    else:
        raise ValueError(f"Analysis method {type(analysis_method)} doesn't support prediction")
    
    # Convert to cluster membership matrix
    n_clusters = analysis_method.get_n_components()
    patch_components = np.zeros((len(patch_features), n_clusters))
    
    if cluster_assignments.ndim == 1:
        # Hard assignments (K-means)
        for i, cluster_id in enumerate(cluster_assignments):
            if 0 <= cluster_id < n_clusters:
                patch_components[i, cluster_id] = 1.0
    else:
        # Soft assignments (GMM, PCA components)
        patch_components = cluster_assignments
    
    # Detect all animals using existing detection pipeline
    detections = detect_animals_from_clusters(
        patch_components=patch_components,
        patch_coordinates=patch_coordinates,
        cluster_labels=cluster_labels,
        relative_patch_size=relative_patch_size,
        background_labels=["background", "vegetation", "ground", "shadow"],
        kernel_size=1
    )
    
    # Process each detection individually
    detection_results = []
    width, height = image.size
    
    for i, detection in enumerate(detections):
        # Convert detection bbox to pixel coordinates
        predicted_bbox = (
            int(detection.bbox.x1 * width),   # x_min
            int(detection.bbox.y1 * height),  # y_min  
            int(detection.bbox.x2 * width),   # x_max
            int(detection.bbox.y2 * height)   # y_max
        )
        
        # Validate detection has sufficient patches for viewpoint estimation
        is_valid = validate_detection_patches(
            patch_coordinates, patch_components, detection.bbox, relative_patch_size
        )
        
        # Classify viewpoint for THIS specific detection
        predicted_viewpoint = 'Unknown'
        viewpoint_confidence = 0.0
        
        if depth_map is not None and is_valid:
            # Filter patches to only those within this detection's bbox
            filtered_coords, filtered_components, filtered_depths = filter_patches_for_detection(
                patch_coordinates, patch_components, patch_depths, detection.bbox, relative_patch_size
            )
            
            # Run viewpoint estimation on filtered patches
            if len(filtered_coords) > 0:  # Ensure we have patches to work with
                viewpoint_result = estimate_viewpoint_with_axis_fitting(
                    patch_components=filtered_components,
                    patch_coordinates=filtered_coords,
                    patch_depths=filtered_depths,
                    cluster_labels=cluster_labels,
                    image_size=image.size,
                    forward_threshold=config.get('forward_threshold', 45.0),
                    side_threshold=config.get('side_threshold', 75.0)
                )
                
                if viewpoint_result:
                    predicted_viewpoint = viewpoint_result['viewpoint']
                    viewpoint_confidence = viewpoint_result.get('confidence', 0.0)
        
        detection_results.append({
            'detection_id': i,
            'bbox': predicted_bbox,
            'confidence': detection.confidence,
            'viewpoint': predicted_viewpoint,
            'viewpoint_confidence': viewpoint_confidence,
            'is_valid': is_valid
        })
    
    # Ground truth bounding box and viewpoint
    gt_bbox = (annotation.bbox.x1, annotation.bbox.y1, annotation.bbox.x2, annotation.bbox.y2)
    gt_viewpoint = annotation.viewpoint if hasattr(annotation, 'viewpoint') else 'Unknown'
    
    # Calculate IoU for best detection (for backwards compatibility)
    best_detection_iou = 0.0
    best_detection_bbox = None
    best_detection_viewpoint = 'Unknown'
    best_detection_confidence = 0.0
    
    if detection_results:
        # Find detection with highest IoU to ground truth
        best_iou = 0.0
        best_idx = 0
        
        for i, det_result in enumerate(detection_results):
            iou = calculate_bbox_iou(det_result['bbox'], gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        best_detection = detection_results[best_idx]
        best_detection_iou = best_iou
        best_detection_bbox = best_detection['bbox']
        best_detection_viewpoint = best_detection['viewpoint']
        best_detection_confidence = best_detection['viewpoint_confidence']
    
    return {
        'image_path': str(annotation.image_path),
        'detections': detection_results,  # All detections with individual viewpoints
        'gt_bbox': gt_bbox,
        'gt_viewpoint': gt_viewpoint,
        # Backwards compatibility fields (best detection)
        'predicted_bbox': best_detection_bbox,
        'detection_iou': best_detection_iou,
        'predicted_viewpoint': best_detection_viewpoint,
        'viewpoint_confidence': best_detection_confidence
    }


def evaluate_dataset(config_path: str, 
                    output_dir: str = "outputs/evaluation",
                    subset_size: Optional[int] = None) -> Dict:
    """
    Evaluate detection and viewpoint classification on entire dataset.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save evaluation results
        subset_size: If specified, evaluate only this many images (for testing)
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    # Load configuration - FAIL if not found
    config = load_config(Path(config_path))
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_path / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting dataset evaluation...")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output: {output_dir}")
    
    # Load dataset
    dataset_root = Path(config['dataset']['root_path'])
    crop_to_bbox = config['dataset'].get('crop_to_bbox', False)
    loader = CVATLoader(dataset_root, crop_to_bbox=crop_to_bbox)
    
    # Subset for testing if requested
    annotations = loader.annotations
    if subset_size is not None:
        annotations = annotations[:subset_size]
        logging.info(f"Using subset of {len(annotations)} images")
    
    logging.info(f"Evaluating {len(annotations)} images")
    logging.info(f"Available viewpoints: {loader.viewpoints}")
    
    # Initialize feature extractor
    model_config = config['model']
    extractor = DINOv2PatchExtractor(
        model_name=model_config['dinov2_model'],
        device=model_config['device'],
        image_size=model_config['image_size'],
        enable_depth=model_config.get('enable_depth', False),
        depth_dataset=model_config.get('depth_dataset', 'nyu')
    )
    logging.info(f"Initialized DINOv2 extractor: {model_config['dinov2_model']}")
    
    # Load analysis method - FAIL if not found
    analysis_config = config['analysis']
    model_path = Path(analysis_config['output_dir']) / analysis_config['model_filename']
    analysis_method = load_analysis_model(model_path)
    logging.info(f"Loaded analysis model from {model_path}")
    
    # Load cluster labels from file - FAIL if not found
    cluster_labels_config = config['cluster_labeling']
    labels_file_path = cluster_labels_config['labels_file']
    labels_file_full_path = Path(labels_file_path)
    
    if not labels_file_full_path.is_absolute():
        labels_file_full_path = dataset_root / labels_file_path
    
    with open(labels_file_full_path, 'r') as f:
        cluster_labels = json.load(f)
    
    logging.info(f"Loaded cluster labels from {labels_file_full_path}")
    
    # Process all images
    results = []
    
    for annotation in tqdm(annotations, desc="Evaluating images"):
        result = evaluate_single_image(
            annotation, loader, extractor, analysis_method, cluster_labels, config
        )
        results.append(result)
        
        # Memory cleanup
        gc.collect()
    
    logging.info(f"Completed evaluation: {len(results)} images processed")
    
    # Calculate statistics
    stats = calculate_evaluation_statistics(results, loader.viewpoints)
    
    # Save results
    save_evaluation_results(results, stats, output_path)
    
    # Create visualizations
    create_evaluation_visualizations(results, stats, output_path)
    
    logging.info("Evaluation complete!")
    return {'results': results, 'statistics': stats}


def calculate_evaluation_statistics(results: List[Dict], viewpoint_classes: List[str]) -> Dict:
    """Calculate comprehensive evaluation statistics for multiple detections per image."""
    
    # Extract all individual detections from all images
    all_detections = []
    images_with_detections = 0
    total_detections = 0
    valid_detections = 0
    
    for result in results:
        if result['detections']:
            images_with_detections += 1
            for detection in result['detections']:
                all_detections.append({
                    'image_path': result['image_path'],
                    'detection_id': detection['detection_id'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'viewpoint': detection['viewpoint'],
                    'viewpoint_confidence': detection['viewpoint_confidence'],
                    'is_valid': detection['is_valid'],
                    'gt_viewpoint': result['gt_viewpoint']
                })
                total_detections += 1
                if detection['is_valid']:
                    valid_detections += 1
    
    # Detection statistics (backwards compatible - use best detection per image)
    detection_ious = [r['detection_iou'] for r in results if r['detection_iou'] is not None and r['detection_iou'] > 0]
    
    detection_stats = {
        'mean_iou': np.mean(detection_ious) if detection_ious else 0.0,
        'median_iou': np.median(detection_ious) if detection_ious else 0.0,
        'iou_std': np.std(detection_ious) if detection_ious else 0.0,
        'detection_success_rate': len(detection_ious) / len(results) if results else 0.0,
        'high_iou_rate': sum(1 for iou in detection_ious if iou > 0.5) / len(detection_ious) if detection_ious else 0.0,
        # New multi-detection statistics
        'total_detections': total_detections,
        'valid_detections': valid_detections,
        'images_with_detections': images_with_detections,
        'avg_detections_per_image': total_detections / len(results) if results else 0.0,
        'detection_validity_rate': valid_detections / total_detections if total_detections > 0 else 0.0
    }
    
    # Viewpoint classification statistics (backwards compatible - use best detection per image)
    gt_viewpoints = [r['gt_viewpoint'] for r in results if r['gt_viewpoint'] != 'Unknown']
    pred_viewpoints = [r['predicted_viewpoint'] for r in results if r['gt_viewpoint'] != 'Unknown']
    
    # Per-detection viewpoint statistics (new)
    valid_detections_with_viewpoint = [
        det for det in all_detections 
        if det['is_valid'] and det['gt_viewpoint'] != 'Unknown' and det['viewpoint'] != 'Unknown'
    ]
    
    det_gt_viewpoints = [det['gt_viewpoint'] for det in valid_detections_with_viewpoint]
    det_pred_viewpoints = [det['viewpoint'] for det in valid_detections_with_viewpoint]
    
    if gt_viewpoints and pred_viewpoints:
        # Classification accuracy (best detection per image)
        correct_predictions = sum(1 for gt, pred in zip(gt_viewpoints, pred_viewpoints) if gt == pred)
        viewpoint_accuracy = correct_predictions / len(gt_viewpoints)
        
        # Per-class statistics
        unique_classes = list(set(gt_viewpoints + pred_viewpoints))
        class_report = classification_report(gt_viewpoints, pred_viewpoints, 
                                           labels=unique_classes, 
                                           output_dict=True, 
                                           zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(gt_viewpoints, pred_viewpoints, labels=unique_classes)
        
        viewpoint_stats = {
            'accuracy': viewpoint_accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_labels': unique_classes,
            'total_samples': len(gt_viewpoints)
        }
    else:
        viewpoint_stats = {
            'accuracy': 0.0,
            'classification_report': {},
            'confusion_matrix': [],
            'class_labels': [],
            'total_samples': 0
        }
    
    # Per-detection viewpoint statistics (new)
    if det_gt_viewpoints and det_pred_viewpoints:
        det_correct = sum(1 for gt, pred in zip(det_gt_viewpoints, det_pred_viewpoints) if gt == pred)
        det_accuracy = det_correct / len(det_gt_viewpoints)
        
        det_unique_classes = list(set(det_gt_viewpoints + det_pred_viewpoints))
        det_class_report = classification_report(det_gt_viewpoints, det_pred_viewpoints,
                                               labels=det_unique_classes,
                                               output_dict=True,
                                               zero_division=0)
        
        per_detection_viewpoint_stats = {
            'accuracy': det_accuracy,
            'classification_report': det_class_report,
            'total_samples': len(det_gt_viewpoints),
            'class_labels': det_unique_classes
        }
    else:
        per_detection_viewpoint_stats = {
            'accuracy': 0.0,
            'classification_report': {},
            'total_samples': 0,
            'class_labels': []
        }
    
    return {
        'detection': detection_stats,
        'viewpoint': viewpoint_stats,  # Best detection per image (backwards compatible)
        'per_detection_viewpoint': per_detection_viewpoint_stats,  # All valid detections
        'overall': {
            'total_images': len(results),
            'successful_evaluations': len(results),
            'success_rate': 1.0
        }
    }


def save_evaluation_results(results: List[Dict], stats: Dict, output_path: Path) -> None:
    """Save evaluation results and statistics to files."""
    # Save detailed results
    results_file = output_path / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save statistics
    stats_file = output_path / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save summary report
    summary_file = output_path / 'summary_report.txt'
    with open(summary_file, 'w') as f:
        f.write("Dataset Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall stats
        overall = stats['overall']
        f.write(f"Total Images: {overall['total_images']}\n")
        f.write(f"Success Rate: {overall['success_rate']:.3f}\n\n")
        
        # Detection stats
        detection = stats['detection']
        f.write("Detection Performance:\n")
        f.write(f"  Mean IoU: {detection['mean_iou']:.3f}\n")
        f.write(f"  Median IoU: {detection['median_iou']:.3f}\n")
        f.write(f"  High IoU Rate (>0.5): {detection['high_iou_rate']:.3f}\n")
        f.write(f"  Detection Success Rate: {detection['detection_success_rate']:.3f}\n\n")
        
        # Viewpoint stats
        viewpoint = stats['viewpoint']
        f.write("Viewpoint Classification Performance:\n")
        f.write(f"  Overall Accuracy: {viewpoint['accuracy']:.3f}\n")
        f.write(f"  Total Samples: {viewpoint['total_samples']}\n")
        
        if viewpoint['classification_report']:
            f.write("\nPer-Class Performance:\n")
            for class_name, metrics in viewpoint['classification_report'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    if isinstance(metrics, dict):
                        f.write(f"  {class_name}: P={metrics.get('precision', 0):.3f}, "
                               f"R={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}\n")


def create_evaluation_visualizations(results: List[Dict], stats: Dict, output_path: Path) -> None:
    """Create evaluation visualizations."""
    # IoU distribution
    plt.figure(figsize=(10, 6))
    ious = [r['detection_iou'] for r in results if r['detection_iou'] is not None]
    if ious:
        plt.hist(ious, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Detection IoU')
        plt.ylabel('Count')
        plt.title('Distribution of Detection IoU Scores')
        plt.axvline(x=0.5, color='red', linestyle='--', label='IoU=0.5 threshold')
        plt.legend()
        plt.savefig(output_path / 'iou_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Confusion matrix
    # if stats['viewpoint']['confusion_matrix'] and stats['viewpoint']['class_labels']:
    #     plt.figure(figsize=(10, 8))
    #     conf_matrix = np.array(stats['viewpoint']['confusion_matrix'])
    #     class_labels = stats['viewpoint']['class_labels']
        
    #     sns.heatmap(conf_matrix, annot=True, fmt='d', 
    #                xticklabels=class_labels, yticklabels=class_labels,
    #                cmap='Blues')
    #     plt.xlabel('Predicted Viewpoint')
    #     plt.ylabel('Ground Truth Viewpoint')
    #     plt.title('Viewpoint Classification Confusion Matrix')
    #     plt.tight_layout()
    #     plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    
    # Viewpoint confidence distribution
    confidences = [r['viewpoint_confidence'] for r in results if r['viewpoint_confidence'] > 0]
    if confidences:
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Viewpoint Classification Confidence')
        plt.ylabel('Count')
        plt.title('Distribution of Viewpoint Classification Confidence')
        plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run dataset evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate detection and viewpoint classification on dataset')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--output', default='outputs/evaluation', help='Output directory')
    parser.add_argument('--subset', type=int, help='Evaluate only subset of images (for testing)')
    
    args = parser.parse_args()
    
    results = evaluate_dataset(
        config_path=args.config,
        output_dir=args.output,
        subset_size=args.subset
    )
    
    print(f"\nEvaluation complete! Results saved to: {args.output}")
    print(f"Overall accuracy: {results['statistics']['viewpoint']['accuracy']:.3f}")
    print(f"Mean detection IoU: {results['statistics']['detection']['mean_iou']:.3f}")


if __name__ == "__main__":
    main()