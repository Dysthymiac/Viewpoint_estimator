"""
Pipeline inference script for COCO dataset.

Runs the animal detection pipeline on all images in a COCO dataset
and generates annotations in the same format as COCOAnnotation.
"""

import json
import uuid
import pickle
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import yaml
import numpy as np

from src.data.coco_loader import COCOLoader, COCOImage, COCOAnnotation
from src.data.cvat_loader import BoundingBox
from src.pipeline.animal_detection_pipeline import create_pipeline_from_main_config

from src.pose.batch_detection_dataset import BatchDetectionDataset

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return []  # obj.tolist()
        elif isinstance(obj, BoundingBox):
            return [obj.x1, obj.y1, obj.x2, obj.y2]
        return super().default(obj)


def create_annotation_from_detection(
    detection: Dict[str, Any],
    image_info: COCOImage,
    detection_idx: int
) -> COCOAnnotation:
    """
    Create a COCOAnnotation from pipeline detection results.
    
    Args:
        detection: Detection dictionary from pipeline
        image_info: Original image metadata
        detection_idx: Index of this detection in the image
    
    Returns:
        COCOAnnotation with estimated viewpoint and bounding box
    """
    # Generate unique UUID for annotation
    annotation_uuid = str(uuid.uuid4())
    
    # Extract viewpoint from detection
    viewpoint = "unknown"
    if 'viewpoint' in detection:
        viewpoint_info = detection['viewpoint']
        if isinstance(viewpoint_info, dict) and 'viewpoint' in viewpoint_info:
            viewpoint = viewpoint_info['viewpoint']
        elif isinstance(viewpoint_info, str):
            viewpoint = viewpoint_info
    
    # Extract bounding box from detection
    bbox = None
    annot_census = False
    
    if 'body_part_box' in detection and detection['body_part_box'] is not None:
        body_part_box = detection['body_part_box']
        bbox = BoundingBox(
            x1=body_part_box.x1,
            y1=body_part_box.y1,
            x2=body_part_box.x2,
            y2=body_part_box.y2
        )
        annot_census = True
    else:
        # Create empty bounding box if no detection
        bbox = BoundingBox(x1=0, y1=0, x2=0, y2=0)
        annot_census = False
    
    return COCOAnnotation(
        uuid=annotation_uuid,
        image_uuid=image_info.uuid,
        bbox=bbox,
        individual_id="",  # Empty as requested
        viewpoint=viewpoint,
        category_id=0,  # Default category as requested
        annot_census=annot_census
    )


def get_processed_images(output_dir: Path) -> set[str]:
    """
    Get set of already processed image UUIDs by checking for individual result files.
    
    Args:
        output_dir: Directory containing individual result files
        
    Returns:
        Set of processed image UUIDs
    """
    processed_images = set()
    
    if output_dir.exists():
        for file_path in output_dir.glob("*.pkl"):
            # Extract UUID from filename (format: {uuid}.pkl)
            image_uuid = file_path.stem
            processed_images.add(image_uuid)
    
    return processed_images


class BatchedResultWriter:
    """
    Accumulates detections and writes them in batches for efficient I/O.
    """
    
    def __init__(self, output_dir: Path, batch_size: int = 1000):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.current_batch = []
        self.batch_number = 0
        self.total_detections = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_image_results(self, image_uuid: str, detections: List[Dict], 
                         image_info: Dict, coco_image: COCOImage) -> None:
        """Add results for one image to the batch."""
        # Prepare detailed detections for batched output
        for detection_idx, detection in enumerate(detections):
            detailed_detection = {
                "image_uuid": image_uuid,
                "detection_index": detection_idx,
                "image_info": image_info,
                **detection
            }
            self.current_batch.append(detailed_detection)
            self.total_detections += 1
        
        # Save batch if it's full
        if len(self.current_batch) >= self.batch_size:
            self._save_current_batch()
    
    def _save_current_batch(self) -> None:
        """Save current batch to file."""
        if not self.current_batch:
            return
            
        batch_file = self.output_dir / f"batch_{self.batch_number:06d}.pkl"
        batch_data = {
            'detailed_detections': self.current_batch,
            'batch_number': self.batch_number,
            'num_detections': len(self.current_batch)
        }
        
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        
        self.current_batch = []
        self.batch_number += 1
    
    def finalize(self) -> None:
        """Save any remaining detections and cleanup."""
        if self.current_batch:
            self._save_current_batch()
        
        print(f"Saved {self.total_detections} detections in {self.batch_number} batches to {self.output_dir}")


def save_image_results(image_uuid: str, detections: List[Dict], image_info: Dict, 
                      coco_image: COCOImage, batch_writer: BatchedResultWriter) -> None:
    """
    Save results for a single image using batched writer.
    
    Args:
        image_uuid: UUID of the image
        detections: List of detection results
        image_info: Image processing info from pipeline
        coco_image: Original COCO image metadata
        batch_writer: BatchedResultWriter instance
    """
    batch_writer.add_image_results(image_uuid, detections, image_info, coco_image)


def collect_all_results(main_config_path: str, output_dir: Path, final_output_path: Path) -> None:
    """
    Collect all batch result files into final JSON output using streaming processing.
    
    Args:
        output_dir: Directory containing batch result files
        final_output_path: Path for final combined JSON output
    """
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    dataset_config = main_config['dataset']
    coco_loader = COCOLoader(
        coco_json_path=Path(dataset_config['annotations_file']),
        dataset_root=Path(dataset_config['root_path']),
        crop_to_bbox=dataset_config.get('crop_to_bbox', False),
        images_dir=dataset_config.get("images_dir", None)
    )

    all_images = {}  # Use dict to deduplicate by image_uuid
    all_annotations = []
    
    print(f"Collecting results from {output_dir}...")
    
    # Load batch dataset
    dataset = BatchDetectionDataset(output_dir)
    
    # Process detections in smaller batches to avoid memory issues
    total_processed = 0
    for detection_batch in dataset.iter_batches(500):  # Smaller batch size
        for detection in detection_batch:
            image_uuid = detection["image_uuid"]
            image_info = detection["image_info"]
            
            # Add unique images
            if image_uuid not in all_images:
                all_images[image_uuid] = image_info
            
            # Create annotation from detection
            if "body_part_box" in detection and detection["body_part_box"] is not None:
                annotation = create_annotation_from_detection(
                    detection, 
                    type('COCOImage', (), {"uuid": image_uuid})(),  # Mock COCOImage
                    detection["detection_index"]
                )
                all_annotations.append(annotation.__dict__)
            
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} detections...")
    
    print(f"Processed {total_processed} total detections")
    
    # Create final output and convert to JSON-serializable format
    output_data = {
        "images": [img.__dict__ for img in coco_loader.images.values()],
        "annotations": all_annotations,
        "categories": [{"id": 0, "species": "estimated"}]
    }
    print("serializing")
    # Convert NumPy arrays to lists for JSON serialization
    # json_serializable_data = convert_numpy_to_json_serializable(output_data)
    print("serialized")
    # Save main results
    with open(final_output_path, 'w') as f:
        json.dump(output_data, f, indent=2, 
                       cls=NumpyEncoder)
    
    print(f"Saved {len(all_annotations)} annotations for {len(all_images)} images to {final_output_path}")
    
    # # Save detailed results
    # detailed_output_path = final_output_path.with_suffix('.detailed.pkl')
    # detailed_data = {"detections": all_detailed_detections}
    
    # with open(detailed_output_path, 'wb') as f:
    #     pickle.dump(detailed_data, f)
    
    # print(f"Saved {len(all_detailed_detections)} detailed detections to {detailed_output_path}")


def run_pipeline_inference(
    main_config_path: str,
    output_path: str,
    max_images: int = None,
    save_detailed: bool = True
) -> None:
    """
    Run pipeline inference on COCO dataset with individual file saving.
    
    Args:
        main_config_path: Path to main configuration file
        output_path: Path to save final combined output JSON
        max_images: Maximum number of images to process (None = all)
        save_detailed: Whether to save detailed detection information
    """
    print(f"Loading pipeline from config: {main_config_path}")
    
    # Create output directory for individual results
    output_path = Path(output_path)
    output_dir = output_path.parent / f"{output_path.stem}_individual"
    
    # Check what has been processed already
    processed_images = get_processed_images(output_dir)
    print(f"Found {len(processed_images)} already processed images")
    
    # Create pipeline from main config
    pipeline = create_pipeline_from_main_config(main_config_path)
    
    # Load COCO dataset from main config
    with open(main_config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    dataset_config = main_config['dataset']
    coco_loader = COCOLoader(
        coco_json_path=Path(dataset_config['annotations_file']),
        dataset_root=Path(dataset_config['root_path']),
        crop_to_bbox=dataset_config.get('crop_to_bbox', False),
        images_dir=dataset_config.get("images_dir", None)
    )
    
    print(f"Loaded dataset with {len(coco_loader)} annotations")
    
    # Get unique images (avoid processing same image multiple times)
    unique_images = {}
    for annotation in coco_loader.filter_by_category("zebra_grevys"):
        if annotation.image_uuid not in unique_images:
            unique_images[annotation.image_uuid] = coco_loader.images[annotation.image_uuid]
    
    print(f"Found {len(unique_images)} unique images")
    
    # Filter out already processed images
    unprocessed_images = {
        uuid: img for uuid, img in unique_images.items() 
        if uuid not in processed_images
    }
    print(f"Found {len(unprocessed_images)} unprocessed images")
    
    # Limit images if specified
    if max_images is not None:
        image_items = list(unprocessed_images.items())[:max_images]
        unprocessed_images = dict(image_items)
        print(f"Processing first {len(unprocessed_images)} unprocessed images")
    
    # Create batched result writer for efficient I/O
    batch_writer = None
    if save_detailed:
        batch_writer = BatchedResultWriter(output_dir, batch_size=1000)
    
    # Process each unprocessed image and save immediately
    processed_count = 0
    
    for i, (image_uuid, coco_image) in enumerate(unprocessed_images.items()):
        print(f"Processing image {i+1}/{len(unprocessed_images)}: {image_uuid}")
        
        try:
            # Load full image
            image_path = coco_loader.get_image_path_from_uuid(image_uuid)
            image = Image.open(image_path).convert('RGB')
            
            # Run pipeline
            detections, image_info = pipeline.process(image)
            
            print(f"  Found {len(detections)} detections")
            
            # Save results for this image immediately
            if save_detailed:
                save_image_results(image_uuid, detections, image_info, coco_image, batch_writer)
                processed_count += 1
                print(f"  Saved results (individual + batch)")
            
        except Exception as e:
            print(f"  Error processing image {image_uuid}: {e}")
    
    # Finalize batched output
    if batch_writer:
        batch_writer.finalize()
    
    print(f"Processed {processed_count} images successfully")
    print(f"Individual results saved to {output_dir}")
    print(f"Use collect_all_results() to combine into final output")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline inference on COCO dataset")
    parser.add_argument("config", help="Path to main configuration file")
    parser.add_argument("output", help="Path to save output annotations JSON")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--no-detailed", action="store_true", help="Skip saving detailed detection information")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing individual results into final output")
    
    args = parser.parse_args()
    
    if args.collect_only:
        # Only collect existing results
        output_path = Path(args.output)
        output_dir = output_path.parent / f"{output_path.stem}_individual"
        collect_all_results(args.config, output_dir, output_path)
    else:
        # Run inference
        run_pipeline_inference(
            main_config_path=args.config,
            output_path=args.output,
            max_images=args.max_images,
            save_detailed=not args.no_detailed
        )


if __name__ == "__main__":
    main()