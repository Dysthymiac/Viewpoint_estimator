"""Main pipeline for viewpoint analysis using DINOv2 and PCA."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to Python path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable to avoid OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.analysis import PipelineMethod
from src.data.cvat_loader import CVATLoader
from src.data.coco_loader import COCOLoader
from src.models.dinov2_extractor import DINOv2PatchExtractor
from src.utils.analysis_utils import (
    create_analysis_method_from_config,
    get_chunk_generator,
    save_analysis_model,
)
from src.utils.config import load_config, setup_directories


def setup_logging(config: dict) -> None:
    """Configure logging."""
    log_config = config['logging']
    log_path = Path(log_config['log_dir']) / log_config['log_file']
    
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def extract_feature_chunks(loader: CVATLoader, extractor: DINOv2PatchExtractor, chunk_size: int = 50):
    """Generator that yields feature chunks without storing everything in memory."""
    total_chunks = (len(loader) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(loader))
        
        chunk_features = []
        chunk_coordinates = []
        chunk_labels = []
        chunk_annotations = loader.annotations[start_idx:end_idx]
        
        for annotation in tqdm(chunk_annotations, desc=f"Chunk {chunk_idx + 1}/{total_chunks}", leave=False):
            image = loader.load_image(annotation)
            patch_features, coordinates, _ = extractor.extract_patch_features(image)
            chunk_features.append(patch_features)
            chunk_coordinates.append(coordinates)
            chunk_labels.append(annotation.viewpoint)
        
        # Stack chunk and yield
        chunk_array = np.vstack(chunk_features) if chunk_features else None
        yield chunk_idx, chunk_array, chunk_coordinates, chunk_labels


def save_chunk(chunk_array: np.ndarray, chunk_coordinates: list, chunk_labels: list, chunk_idx: int, output_dir: Path, config: dict) -> Path:
    """Save a self-contained feature chunk to disk."""
    chunks_dir = output_dir / config['features']['chunks_dir']
    chunks_dir.mkdir(exist_ok=True)
    
    chunk_file = chunks_dir / config['features']['chunk_file_pattern'].format(chunk_idx)
    
    # Save with allow_pickle=True for variable-length coordinate arrays
    np.savez_compressed(chunk_file, 
                       features=chunk_array,
                       coordinates=np.array(chunk_coordinates, dtype=object), 
                       labels=np.array(chunk_labels, dtype=object))
    logging.info(f"Chunk {chunk_idx} saved: {chunk_array.shape} -> {chunk_file}")
    return chunk_file


def load_chunk(chunk_file: Path) -> tuple[np.ndarray, list, list]:
    """Load a self-contained chunk from disk."""
    data = np.load(chunk_file, allow_pickle=True)
    coordinates = data['coordinates'].tolist() if data['coordinates'].dtype == object else data['coordinates']
    labels = data['labels'].tolist() if data['labels'].dtype == object else data['labels']
    return data['features'], coordinates, labels


def process_features_chunked(loader: CVATLoader, extractor: DINOv2PatchExtractor, config: dict) -> tuple[list[np.ndarray], list[str]]:
    """Process features in chunks: extract -> save -> fit analysis method incrementally."""
    
    output_dir = Path(config['features']['output_dir'])
    chunks_dir = output_dir / config['features']['chunks_dir']
    chunks_dir.mkdir(exist_ok=True)
    analysis_config = config['analysis']
    
    # Initialize analysis method
    analysis_method = create_analysis_method_from_config(analysis_config)
    all_coordinates = []
    all_labels = []
    
    logging.info(f"Processing {len(loader)} images in chunks...")
    logging.info(f"Analysis method: {analysis_method.get_method_name()}")
    logging.info(f"Components: {analysis_method.get_n_components()}")
    logging.info(f"Supports incremental fit: {analysis_method.supports_incremental_fit()}")
    
    # Process chunks with incremental loading/extraction
    chunk_size = 50
    total_chunks = (len(loader) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(total_chunks):
        # if chunk_idx > 30:
        #     break  # Limit to first 30 chunks for testing
        chunk_file = chunks_dir / config['features']['chunk_file_pattern'].format(chunk_idx)
        
        # Check if chunk already exists
        if chunk_file.exists():
            logging.info(f"Loading existing chunk {chunk_idx + 1}/{total_chunks}: {chunk_file}")
            chunk_features, chunk_coordinates, chunk_labels = load_chunk(chunk_file)
        else:
            logging.info(f"Extracting chunk {chunk_idx + 1}/{total_chunks}: images {chunk_idx * chunk_size} to {min((chunk_idx + 1) * chunk_size, len(loader)) - 1}")
            
            # Extract features for this chunk
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(loader))
            chunk_annotations = loader.annotations[start_idx:end_idx]
            
            chunk_features_list = []
            chunk_coordinates = []
            chunk_labels = []
            
            for annotation in tqdm(chunk_annotations, desc=f"Chunk {chunk_idx + 1}/{total_chunks}", leave=False):
                image = loader.load_image(annotation)
                patch_features, coordinates, _ = extractor.extract_patch_features(image)
                chunk_features_list.append(patch_features)
                chunk_coordinates.append(coordinates)
                chunk_labels.append(annotation.viewpoint)
            
            # Stack chunk features
            chunk_features = np.vstack(chunk_features_list) if chunk_features_list else None
            
            if chunk_features is not None:
                # Save chunk
                save_chunk(chunk_features, chunk_coordinates, chunk_labels, chunk_idx, output_dir, config)
                logging.info(f"Saved chunk {chunk_idx}: {chunk_features.shape}")
            
            # Clean up extraction memory
            del chunk_features_list
            gc.collect()
        
        # Fit analysis method incrementally (regardless of whether chunk was loaded or extracted)
        if chunk_features is not None:
            if analysis_method.supports_incremental_fit():
                analysis_method.incremental_fit(chunk_features)
                logging.info(f"Incremental fit on chunk {chunk_idx}: {chunk_features.shape}")
            else:
                logging.warning(f"Method doesn't support incremental fit. Skipping chunk {chunk_idx}")
            
            # Collect metadata incrementally
            all_coordinates.extend(chunk_coordinates)
            all_labels.extend(chunk_labels)
            
            # Clean up memory
            del chunk_features
            gc.collect()
    
    # Finalize pipeline if needed
    if isinstance(analysis_method, PipelineMethod):
        logging.info("Finalizing pipeline...")
        chunk_generator = get_chunk_generator(chunks_dir, config['features']['chunk_file_pattern'])
        analysis_method.finalize_pipeline(chunk_generator)
        logging.info("Pipeline finalization complete")
    
    # Save analysis model
    if analysis_config.get('save_model', True):
        model_path = Path(analysis_config['output_dir']) / analysis_config['model_filename']
        save_analysis_model(analysis_method, model_path)
        logging.info(f"Analysis model saved to {model_path}")
    
    # Log method-specific metadata
    metadata = analysis_method.get_metadata()
    if metadata:
        logging.info(f"Analysis complete. Metadata: {metadata}")
    
    logging.info(f"Chunked processing complete. Method: {analysis_method.get_method_name()}")
    return all_coordinates, all_labels


def run_patch_pca_analysis(
    all_coordinates: list[np.ndarray], 
    labels: list[str], 
    loader: CVATLoader, 
    config: dict
) -> None:
    """Create visualizations using pre-fitted PCA model and chunked data."""
    pca_config = config['pca']
    viz_config = config['visualization']
    
    # Load the fitted PCA model
    analyzer = PatchPCAAnalyzer(n_components=pca_config['n_components'])
    pca_path = Path(pca_config['output_dir']) / "patch_pca_model.npz"
    
    if not pca_path.exists():
        logging.error(f"PCA model not found at {pca_path}. Run feature extraction first.")
        return
        
    analyzer.load_pca_model(pca_path)
    logging.info(f"Loaded PCA model from {pca_path}")
    
    # Transform chunks and collect results
    logging.info("Transforming chunks using fitted PCA...")
    chunks_dir = Path(config['features']['output_dir']) / "feature_chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
    
    all_transformed = []
    for chunk_file in chunk_files:
        chunk_features, _, _ = load_chunk(chunk_file)
        transformed_chunk = analyzer.transform(chunk_features)
        all_transformed.append(transformed_chunk)
        logging.debug(f"Transformed {chunk_file}: {chunk_features.shape} -> {transformed_chunk.shape}")
    
    # Stack transformed results
    transformed_patches = np.vstack(all_transformed)
    logging.info(f"Final transformed shape: {transformed_patches.shape}")
    
    # Create patch scatter plot
    scatter_path = Path(viz_config['output_dir']) / "patch_pca_scatter.png"
    analyzer.create_patch_scatter_plot(transformed_patches, scatter_path)
    logging.info(f"Patch PCA scatter plot saved to {scatter_path}")
    
    # Create RGB color mapping for patches
    logging.info("Creating patch-level RGB visualizations...")
    rgb_colors = analyzer.patches_to_rgb(transformed_patches)
    
    # Load sample images for visualization
    sample_images = [loader.load_image(ann) for ann in loader.annotations]
    
    # Create patch-level RGB overlay visualization
    overlay_path = Path(viz_config['output_dir']) / "patch_rgb_overlay.png"
    analyzer.visualize_patch_rgb_overlay(
        sample_images, all_coordinates, rgb_colors, labels, 
        overlay_path, viz_config['n_sample_images']
    )
    logging.info(f"Patch RGB overlay visualization saved to {overlay_path}")
    
    # Create viewpoint comparison
    comparison_path = Path(viz_config['output_dir']) / "viewpoint_comparison.png"
    analyzer.create_viewpoint_comparison(
        sample_images, all_coordinates, rgb_colors, labels, comparison_path
    )
    logging.info(f"Viewpoint comparison saved to {comparison_path}")
    
    # Log explained variance
    explained_var = analyzer.pca.explained_variance_ratio_
    logging.info(f"Explained variance ratios: {explained_var}")
    logging.info(f"Total explained variance: {explained_var.sum():.3f}")
    
    # Log patch statistics
    patches_per_image = len(all_coordinates[0])
    total_patches = len(transformed_patches)
    logging.info(f"Analyzed {total_patches} patches from {len(sample_images)} images")
    logging.info(f"Patches per image: {patches_per_image}")


def create_data_loader(config: dict):
    """Create appropriate data loader based on dataset configuration."""
    dataset_config = config['dataset']
    dataset_root = Path(dataset_config['root_path'])
    crop_to_bbox = dataset_config.get('crop_to_bbox', True)
    dataset_format = dataset_config.get('format', 'cvat')  # Default to CVAT for backward compatibility
    
    # Use images_dir to override root_path if provided
    if 'images_dir' in dataset_config and dataset_config['images_dir']:
        image_root = Path(dataset_config['images_dir'])
        if not image_root.is_absolute():
            image_root = dataset_root / dataset_config['images_dir']
    else:
        image_root = dataset_root
    
    if dataset_format.lower() == 'coco':
        annotations_file = dataset_config['annotations_file']
        coco_json_path = dataset_root / annotations_file
        loader = COCOLoader(image_root, coco_json_path, crop_to_bbox=crop_to_bbox)
        logging.info(f"Loaded COCO dataset: {len(loader)} annotations")
        logging.info(f"Annotations from: {coco_json_path}")
        logging.info(f"Images from: {image_root}")
    elif dataset_format.lower() == 'cvat':
        loader = CVATLoader(dataset_root, crop_to_bbox=crop_to_bbox)
        logging.info(f"Loaded CVAT dataset: {len(loader)} annotations")
        logging.info(f"Dataset root: {dataset_root}")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}. Supported formats: 'cvat', 'coco'")
    
    logging.info(f"Unique viewpoints: {sorted(loader.viewpoints)}")
    logging.info(f"Using {'cropped' if crop_to_bbox else 'full'} images")
    
    return loader


def main() -> None:
    """Main execution pipeline."""
    # Load configuration
    config = load_config(Path("config_zebra_test.yaml"))
    setup_directories(config)
    setup_logging(config)
    
    logging.info("Starting viewpoint analysis pipeline...")
    
    # Load data
    loader = create_data_loader(config)
    
    # Initialize patch feature extractor
    model_config = config['model']
    extractor = DINOv2PatchExtractor(
        model_name=model_config['dinov2_model'],
        device=model_config['device'],
        image_size=model_config['image_size'],
        enable_depth=model_config.get('enable_depth', False),
        depth_dataset=model_config.get('depth_dataset', 'nyu')
    )
    logging.info(f"Initialized DINOv2 patch extractor: {model_config['dinov2_model']}")
    
    # Process features in chunks and fit PCA incrementally
    all_coordinates, labels = process_features_chunked(loader, extractor, config)
    
    # Skip visualization for now - focusing on core functionality only
    
    logging.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()