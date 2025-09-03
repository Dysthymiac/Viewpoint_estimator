"""
Evaluation script for pipeline output against ground truth annotations.

Evaluates algorithm predictions by matching them to ground truth annotations
based on IoU overlap and comparing bounding boxes and viewpoints.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.data.coco_loader import COCOLoader, COCOAnnotation
from src.data.cvat_loader import BoundingBox


@dataclass
class EvaluationMatch:
    """Represents a match between GT and predicted annotation."""
    gt_annotation: COCOAnnotation
    pred_annotation: COCOAnnotation
    iou: float
    viewpoint_match: bool


@dataclass
class ImageEvaluation:
    """Per-image evaluation results."""
    image_uuid: str
    matches: List[EvaluationMatch]
    missed_gt: List[COCOAnnotation]  # GT annotations with no match
    extra_pred: List[COCOAnnotation]  # Predicted annotations with no match
    
    @property
    def num_gt(self) -> int:
        return len(self.matches) + len(self.missed_gt)
    
    @property
    def num_pred(self) -> int:
        return len(self.matches) + len(self.extra_pred)
    
    @property
    def num_matched(self) -> int:
        return len(self.matches)
    
    @property
    def precision(self) -> float:
        return self.num_matched / self.num_pred if self.num_pred > 0 else 0.0
    
    @property
    def recall(self) -> float:
        return self.num_matched / self.num_gt if self.num_gt > 0 else 0.0
    
    @property
    def viewpoint_accuracy(self) -> float:
        if not self.matches:
            return 0.0
        correct = sum(1 for m in self.matches if m.viewpoint_match)
        return correct / len(self.matches)
    
    @property
    def mean_iou(self) -> float:
        if not self.matches:
            return 0.0
        return np.mean([m.iou for m in self.matches])


@dataclass
class SummaryEvaluation:
    """Overall evaluation summary."""
    image_evaluations: List[ImageEvaluation]
    
    @property
    def total_gt(self) -> int:
        return sum(img.num_gt for img in self.image_evaluations)
    
    @property
    def total_pred(self) -> int:
        return sum(img.num_pred for img in self.image_evaluations)
    
    @property
    def total_matched(self) -> int:
        return sum(img.num_matched for img in self.image_evaluations)
    
    @property
    def overall_precision(self) -> float:
        return self.total_matched / self.total_pred if self.total_pred > 0 else 0.0
    
    @property
    def overall_recall(self) -> float:
        return self.total_matched / self.total_gt if self.total_gt > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.overall_precision, self.overall_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def overall_viewpoint_accuracy(self) -> float:
        all_matches = [m for img in self.image_evaluations for m in img.matches]
        if not all_matches:
            return 0.0
        correct = sum(1 for m in all_matches if m.viewpoint_match)
        return correct / len(all_matches)
    
    @property
    def mean_iou(self) -> float:
        all_matches = [m for img in self.image_evaluations for m in img.matches]
        if not all_matches:
            return 0.0
        return np.mean([m.iou for m in all_matches])


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # Calculate intersection
    x_left = max(bbox1.x1, bbox2.x1)
    y_top = max(bbox1.y1, bbox2.y1)
    x_right = min(bbox1.x2, bbox2.x2)
    y_bottom = min(bbox1.y2, bbox2.y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    bbox1_area = bbox1.area()
    bbox2_area = bbox2.area()
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def normalize_viewpoint(viewpoint: str) -> str:
    """Normalize viewpoint string for comparison."""
    return viewpoint.lower().replace(" ", "").replace("_", "")


def match_viewpoints(gt_viewpoint: str, pred_viewpoint: str) -> bool:
    """Check if viewpoints match after normalization."""
    return normalize_viewpoint(gt_viewpoint) == normalize_viewpoint(pred_viewpoint)


def load_predictions(pred_json_path: str) -> Dict[str, List[COCOAnnotation]]:
    """Load predicted annotations from JSON file."""
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    # Create image lookup for dimensions
    images = {img['uuid']: img for img in pred_data['images']}
    
    # Group predictions by image UUID
    predictions = {}
    
    for ann_data in pred_data['annotations']:
        # Skip annotations with empty bounding boxes (all zeros)
        rel_x1, rel_y1, rel_x2, rel_y2 = ann_data['bbox']
        if all(coord == 0 for coord in [rel_x1, rel_y1, rel_x2, rel_y2]):
            continue
        
        # Calculate relative area (since coordinates are relative)
        rel_width = abs(rel_x2 - rel_x1)
        rel_height = abs(rel_y2 - rel_y1)
        rel_area = rel_width * rel_height
        
        # Skip bounding boxes that are too small or too large
        if rel_area < 0.01 or rel_area > 0.8:
            continue
        
        # Get image dimensions for coordinate conversion
        image_info = images[ann_data['image_uuid']]
        width, height = image_info['width'], image_info['height']
        
        # Convert relative coordinates to absolute
        bbox = BoundingBox(
            x1=rel_x1 * width,
            y1=rel_y1 * height,
            x2=rel_x2 * width,
            y2=rel_y2 * height
        )
        
        annotation = COCOAnnotation(
            uuid=ann_data['uuid'],
            image_uuid=ann_data['image_uuid'],
            bbox=bbox,
            viewpoint=ann_data['viewpoint'],
            individual_id=ann_data.get('individual_id', ''),
            category_id=ann_data.get('category_id', 0),
            annot_census=ann_data.get('annot_census', False)
        )
        
        if annotation.image_uuid not in predictions:
            predictions[annotation.image_uuid] = []
        predictions[annotation.image_uuid].append(annotation)
    
    return predictions


def evaluate_image(
    image_uuid: str,
    gt_annotations: List[COCOAnnotation],
    pred_annotations: List[COCOAnnotation],
    iou_threshold: float = 0.25
) -> ImageEvaluation:
    """Evaluate predictions for a single image."""
    matches = []
    used_pred_indices = set()
    
    # For each GT annotation, find the best matching prediction
    for gt_ann in gt_annotations:
        best_match = None
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred_ann in enumerate(pred_annotations):
            if pred_idx in used_pred_indices:
                continue
                
            iou = calculate_iou(gt_ann.bbox, pred_ann.bbox)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = pred_ann
                best_pred_idx = pred_idx
        
        if best_match is not None:
            viewpoint_match = match_viewpoints(gt_ann.viewpoint, best_match.viewpoint)
            matches.append(EvaluationMatch(
                gt_annotation=gt_ann,
                pred_annotation=best_match,
                iou=best_iou,
                viewpoint_match=viewpoint_match
            ))
            used_pred_indices.add(best_pred_idx)
    
    # Identify missed GT and extra predictions
    matched_gt = {m.gt_annotation.uuid for m in matches}
    missed_gt = [ann for ann in gt_annotations if ann.uuid not in matched_gt]
    
    extra_pred = [
        pred_annotations[i] for i in range(len(pred_annotations))
        if i not in used_pred_indices
    ]
    
    return ImageEvaluation(
        image_uuid=image_uuid,
        matches=matches,
        missed_gt=missed_gt,
        extra_pred=extra_pred
    )


def evaluate_pipeline_output(
    gt_coco_path: str,
    pred_json_path: str,
    dataset_root: str = None,
    iou_threshold: float = 0.5,
    category_filter: List[str] = ["zebra_grevys"],
    annot_census: bool = None,
    annot_census_region: bool = None,
    annot_manual: bool = None
) -> SummaryEvaluation:
    """
    Evaluate pipeline output against ground truth.
    
    Args:
        gt_coco_path: Path to ground truth COCO JSON
        pred_json_path: Path to predicted annotations JSON
        dataset_root: Dataset root directory
        iou_threshold: IoU threshold for matching
        category_filter: List of categories to include (default: all)
        annot_census: Filter by annot_census value (None = no filter)
        annot_census_region: Filter by annot_census_region value (None = no filter) 
        annot_manual: Filter by annot_manual value (None = no filter)
        
    Returns:
        SummaryEvaluation with detailed results
    """
    # Load ground truth
    coco_loader = COCOLoader(
        coco_json_path=Path(gt_coco_path),
        dataset_root=Path(dataset_root) if dataset_root else None,
        crop_to_bbox=False
    )
    
    # Filter by category if specified
    if category_filter:
        gt_annotations = coco_loader.filter_by_category(category_filter)
    else:
        gt_annotations = coco_loader.annotations
    
    # Apply annotation filters
    if annot_census is not None:
        gt_annotations = [ann for ann in gt_annotations if ann.annot_census == annot_census]
    
    if annot_census_region is not None:
        gt_annotations = [ann for ann in gt_annotations if ann.annot_census_region == annot_census_region]
    
    if annot_manual is not None:
        gt_annotations = [ann for ann in gt_annotations if ann.annot_manual == annot_manual]
    
    # Group GT annotations by image UUID
    gt_by_image = {}
    for ann in gt_annotations:
        if ann.image_uuid not in gt_by_image:
            gt_by_image[ann.image_uuid] = []
        gt_by_image[ann.image_uuid].append(ann)
    
    # Load predictions
    predictions = load_predictions(pred_json_path)
    
    # Evaluate each image
    image_evaluations = []
    
    for image_uuid in gt_by_image:
        gt_anns = gt_by_image[image_uuid]
        pred_anns = predictions.get(image_uuid, [])
        
        img_eval = evaluate_image(image_uuid, gt_anns, pred_anns, iou_threshold)
        image_evaluations.append(img_eval)
    
    # Also check for predictions on images not in GT
    for image_uuid in predictions:
        if image_uuid not in gt_by_image:
            # All predictions are extra for this image
            pred_anns = predictions[image_uuid]
            img_eval = ImageEvaluation(
                image_uuid=image_uuid,
                matches=[],
                missed_gt=[],
                extra_pred=pred_anns
            )
            image_evaluations.append(img_eval)
    
    return SummaryEvaluation(image_evaluations)


def generate_report(
    evaluation: SummaryEvaluation,
    output_path: str = None,
    filters_used: Dict = None
) -> str:
    """Generate detailed evaluation report."""
    report_lines = []
    
    # Header and filters used
    report_lines.extend([
        "=== PIPELINE EVALUATION REPORT ===\n"
    ])
    
    # Show filters that were applied
    if filters_used:
        report_lines.append("FILTERS APPLIED:")
        if filters_used.get('category_filter'):
            report_lines.append(f"  Categories: {', '.join(filters_used['category_filter'])}")
        if filters_used.get('annot_census') is not None:
            report_lines.append(f"  annot_census: {filters_used['annot_census']}")
        if filters_used.get('annot_census_region') is not None:
            report_lines.append(f"  annot_census_region: {filters_used['annot_census_region']}")
        if filters_used.get('annot_manual') is not None:
            report_lines.append(f"  annot_manual: {filters_used['annot_manual']}")
        if filters_used.get('iou_threshold') is not None:
            report_lines.append(f"  IoU threshold: {filters_used['iou_threshold']}")
        report_lines.append("")
    
    # Summary statistics
    report_lines.extend([
        f"Total GT annotations: {evaluation.total_gt}",
        f"Total predicted annotations: {evaluation.total_pred}",
        f"Total matched annotations: {evaluation.total_matched}",
        "",
        f"Overall Precision: {evaluation.overall_precision:.3f}",
        f"Overall Recall: {evaluation.overall_recall:.3f}",
        f"F1 Score: {evaluation.f1_score:.3f}",
        f"Mean IoU: {evaluation.mean_iou:.3f}",
        f"Viewpoint Accuracy: {evaluation.overall_viewpoint_accuracy:.3f}",
        "",
        f"Images evaluated: {len(evaluation.image_evaluations)}",
        ""
    ])
    
    # Per-image statistics
    report_lines.append("=== PER-IMAGE RESULTS ===\n")
    
    for img_eval in evaluation.image_evaluations:
        report_lines.extend([
            f"Image: {img_eval.image_uuid}",
            f"  GT: {img_eval.num_gt}, Pred: {img_eval.num_pred}, Matched: {img_eval.num_matched}",
            f"  Precision: {img_eval.precision:.3f}, Recall: {img_eval.recall:.3f}",
            f"  Mean IoU: {img_eval.mean_iou:.3f}, Viewpoint Acc: {img_eval.viewpoint_accuracy:.3f}",
            f"  Missed GT: {len(img_eval.missed_gt)}, Extra Pred: {len(img_eval.extra_pred)}",
            ""
        ])
    
    report_text = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to {output_path}")
    
    return report_text


def main():
    """Main entry point."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against ground truth")
    parser.add_argument("config", help="Path to main configuration file")
    parser.add_argument("predictions", help="Path to predicted annotations JSON file")
    parser.add_argument("--output", help="Path to save evaluation report")
    parser.add_argument("--iou-threshold", type=float, default=0.25, help="IoU threshold for matching")
    parser.add_argument("--category", nargs="+", help="Category names to filter (default: from config)")
    parser.add_argument("--annot-census", choices=['true', 'false'], help="Filter by annot_census value")
    parser.add_argument("--annot-census-region", choices=['true', 'false'], help="Filter by annot_census_region value") 
    parser.add_argument("--annot-manual", choices=['true', 'false'], help="Filter by annot_manual value")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract dataset configuration
    dataset_config = config['dataset']
    gt_coco_path = dataset_config['annotations_file']
    dataset_root = dataset_config['root_path']
    
    # Use category filter from args or default to zebra_grevys
    category_filter = args.category if args.category else ["zebra_grevys"]
    
    # Parse boolean filters
    annot_census_filter = args.annot_census == 'true' if args.annot_census else None
    annot_census_region_filter = args.annot_census_region == 'true' if args.annot_census_region else None
    annot_manual_filter = args.annot_manual == 'true' if args.annot_manual else None
    
    # Run evaluation
    evaluation = evaluate_pipeline_output(
        gt_coco_path=gt_coco_path,
        pred_json_path=args.predictions,
        dataset_root=dataset_root,
        iou_threshold=args.iou_threshold,
        category_filter=category_filter,
        annot_census=annot_census_filter,
        annot_census_region=annot_census_region_filter,
        annot_manual=annot_manual_filter
    )
    
    # Collect filters used for report
    filters_used = {
        'category_filter': category_filter,
        'iou_threshold': args.iou_threshold
    }
    
    if annot_census_filter is not None:
        filters_used['annot_census'] = annot_census_filter
    if annot_census_region_filter is not None:
        filters_used['annot_census_region'] = annot_census_region_filter  
    if annot_manual_filter is not None:
        filters_used['annot_manual'] = annot_manual_filter
    
    # Generate and display report
    report = generate_report(evaluation, args.output, filters_used)
    print(report)


if __name__ == "__main__":
    main()