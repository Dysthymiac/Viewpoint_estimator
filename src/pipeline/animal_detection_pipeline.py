"""
Animal Detection and Viewpoint Estimation Pipeline

A simple pipeline class that combines DINOv2 patch extraction, SAM2 segmentation,
and 3D viewpoint estimation with configurable parameters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

from src.utils.analysis_utils import load_analysis_model

from ..models.dinov2_extractor import DINOv2PatchExtractor, aggregate_depth_to_patches
from ..detection.animal_detector import detect_animals_with_sam2, get_detection_body_part_boxes
from ..classification.viewpoint_3d import estimate_viewpoint_for_detections
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    # Detection parameters
    detection: Dict[str, Any] = field(default_factory=lambda: {
        'score_threshold': 0.8,
        'merge_overlap_threshold': 0.5,
        'kernel_size': 1,
        'body_part_labels': ["body", "head", "neck", "back", "belly", "thigh"],
        'background_labels': ["background", "vegetation", "ground", "shadow"],
        'background_threshold': 0.6,
        'image_resize': None,
        'plot_debug': False
    })
    
    # Viewpoint estimation parameters
    viewpoint: Dict[str, Any] = field(default_factory=lambda: {
        'forward_threshold': 45.0,
        'side_threshold': 75.0,
        'depth_mult': 50.0,
        'morphology': True,
        'operations': ["opening", "closing"],
        'kernel_size': 1,
        'kernel_shape': "square"
    })
    
    # Bounding box parameters
    bounding_box: Dict[str, Any] = field(default_factory=lambda: {
        'body_parts': ["head", "neck", "body", "back", "belly"],
        'padding': 0.0,
        'apply_morphology': True,
        'kernel_size': 1
    })


class AnimalDetectionPipeline:
    """
    Simple pipeline for animal detection and viewpoint estimation.
    
    Processes detections through the pipeline, progressively enriching them with:
    1. DINOv2 patch extraction and clustering
    2. SAM2-based instance segmentation
    3. 3D viewpoint estimation
    4. Bounding box extraction
    """
    
    def __init__(
        self,
        depth_extractor: DINOv2PatchExtractor,
        sam2_predictor: SAM2ImagePredictor,
        analysis_method: Any,
        cluster_labels: Dict[str, List[int]],
        config: PipelineConfig
    ):
        """
        Initialize the pipeline.
        
        Args:
            depth_extractor: DINOv2 patch extractor
            sam2_predictor: SAM2 predictor
            analysis_method: Analysis method (PCA, etc.)
            cluster_labels: Body part cluster labels
            config: Pipeline configuration
        """
        self.depth_extractor = depth_extractor
        self.sam2_predictor = sam2_predictor
        self.analysis_method = analysis_method
        self.cluster_labels = cluster_labels
        self.config = config
    
    def process(self, image: Image.Image) -> List[Dict]:
        """
        Run the complete pipeline and return enriched detections.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of detection dictionaries progressively enriched with:
            - SAM2 masks and scores
            - Viewpoint estimation results
            - Body part bounding boxes
        """
        # Extract features and patches
        patch_features, patch_coordinates, relative_patch_size, depthmap = self.depth_extractor.extract_patch_features(image)
        patch_depths = aggregate_depth_to_patches(depthmap, patch_coordinates, relative_patch_size, 'median')

        # # Transform with analysis method
        raw_output = self.analysis_method.transform(patch_features)
        if raw_output.ndim == 1:
            patch_components = np.zeros((len(raw_output), self.analysis_method.get_n_components()))
            patch_components[np.arange(len(raw_output)), raw_output] = 1.0
        else:
            patch_components = raw_output
        
        # Step 1: Detect animals with SAM2
        detections = detect_animals_with_sam2(
            image=image,
            patch_components=patch_components,
            patch_coordinates=patch_coordinates,
            cluster_labels=self.cluster_labels,
            relative_patch_size=self.depth_extractor.relative_patch_size,
            sam2_predictor=self.sam2_predictor,
            **self.config.detection
        )
        
        if len(detections) == 0:
            return detections
        
        # Step 2: Add viewpoint estimation
        detections = estimate_viewpoint_for_detections(
            detections=detections,
            patch_components=patch_components,
            patch_coordinates=patch_coordinates,
            patch_depths=patch_depths,
            relative_patch_size=self.depth_extractor.relative_patch_size,
            cluster_labels=self.cluster_labels,
            image_size=image.size,
            **self.config.viewpoint
        )
        
        # Step 3: Add bounding boxes
        detections = get_detection_body_part_boxes(
            detections=detections,
            patch_components=patch_components,
            patch_coordinates=patch_coordinates,
            relative_patch_size=self.depth_extractor.relative_patch_size,
            cluster_labels=self.cluster_labels,
            image_size=image.size,
            **self.config.bounding_box
        )
        
        image_info = {"patch_components": patch_components,
                    "patch_coordinates": patch_coordinates,
                    "patch_depths": patch_depths,
                    "cluster_labels": self.cluster_labels,
                    "relative_patch_size": self.depth_extractor.relative_patch_size}

        return detections, image_info


def create_pipeline_from_config_file(
    config_path: str,
    depth_extractor: DINOv2PatchExtractor,
    sam2_predictor: SAM2ImagePredictor,
    analysis_method: Any,
    cluster_labels: Dict[str, List[int]]
) -> AnimalDetectionPipeline:
    """
    Create pipeline from configuration file.
    
    Args:
        config_path: Path to YAML/JSON configuration file
        depth_extractor: DINOv2 patch extractor
        sam2_predictor: SAM2 predictor
        analysis_method: Analysis method for clustering
        cluster_labels: Body part cluster labels
        
    Returns:
        Configured AnimalDetectionPipeline instance
    """
    import yaml
    import json
    from pathlib import Path
    
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    config = PipelineConfig(
        detection=config_dict.get('detection', {}),
        viewpoint=config_dict.get('viewpoint', {}),
        bounding_box=config_dict.get('bounding_box', {})
    )
    
    return AnimalDetectionPipeline(
        depth_extractor=depth_extractor,
        sam2_predictor=sam2_predictor,
        analysis_method=analysis_method,
        cluster_labels=cluster_labels,
        config=config
    )


def create_pipeline_from_main_config(main_config_path: str) -> AnimalDetectionPipeline:
    """
    Create pipeline from main configuration file.
    
    Args:
        main_config_path: Path to main YAML configuration file
        
    Returns:
        Configured AnimalDetectionPipeline instance with all components
    """
    import yaml
    import json
    from pathlib import Path
    
    # Load main config
    main_config_path = Path(main_config_path)
    
    if main_config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)
    elif main_config_path.suffix.lower() == '.json':
        with open(main_config_path, 'r') as f:
            main_config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {main_config_path.suffix}")
    
    # Get pipeline config path
    pipeline_config_path = main_config.get('pipeline', {}).get('config_path')
    if not pipeline_config_path:
        raise ValueError("Main config must contain 'pipeline.config_path' field")
    
    # Make path relative to main config directory
    if not Path(pipeline_config_path).is_absolute():
        pipeline_config_path = main_config_path.parent / pipeline_config_path
    
    # Initialize DINOv2 extractor
    model_config = main_config.get('model', {})
    depth_extractor = DINOv2PatchExtractor(
        model_name=model_config['dinov2_model'],
        device=model_config['device'],
        image_size=model_config['image_size'],
        enable_depth=True,
        depth_dataset="nyu",  # Use NYU dataset for indoor/general scenes
        enable_segmentation=False,
        segmentation_head_type="ms"
    )
    
    # Initialize SAM2 predictor
    sam_config = main_config.get('sam', {})
    sam2_predictor = SAM2ImagePredictor.from_pretrained(
        sam_config.get('model', 'facebook/sam2.1-hiera-base-plus')
    )
    
    # Load analysis method (this would need to be loaded from saved model)
    analysis_config = main_config.get('analysis', {})
    model_path = Path(analysis_config['output_dir']) / analysis_config['model_filename']
    analysis_method = load_analysis_model(model_path)
    
    # Extract cluster labels from cluster_labeling config
    cluster_labeling_config = main_config.get('cluster_labeling', {})
    labels_file = Path(cluster_labeling_config.get("labels_file", "./cluster_labels.json"))

    with open(labels_file, 'r') as f:
        cluster_labels = json.load(f)
        print(f"Cluster labels loaded from {labels_file}")
    
    return create_pipeline_from_config_file(
        config_path=str(pipeline_config_path),
        depth_extractor=depth_extractor,
        sam2_predictor=sam2_predictor,
        analysis_method=analysis_method,
        cluster_labels=cluster_labels
    )