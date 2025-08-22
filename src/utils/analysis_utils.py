"""Utilities for saving and loading analysis models."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..analysis import PatchAnalysisMethod, create_method_from_config


def save_analysis_model(method: PatchAnalysisMethod, model_path: Path) -> None:
    """
    Save fitted analysis method to disk.
    
    Args:
        method: Fitted analysis method
        model_path: Path to save model
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_data = {
        'method_name': method.get_method_name(),
        'method_class': method.__class__.__name__,
        'n_components': method.get_n_components(),
        'metadata': method.get_metadata(),
        'method_object': method  # Pickle the entire method object
    }
    
    # Save as compressed npz with pickle for complex objects
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)


def load_analysis_model(model_path: Path) -> PatchAnalysisMethod:
    """
    Load analysis method from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded analysis method
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Analysis model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)
    
    return save_data['method_object']


def create_analysis_method_from_config(config: Dict[str, Any]) -> PatchAnalysisMethod:
    """
    Create analysis method from configuration.
    
    Args:
        config: Analysis configuration dictionary
        
    Returns:
        Analysis method instance
    """
    # Check if using predefined config
    if 'method_config' in config:
        from ..analysis import create_common_method
        return create_common_method(config['method_config'])
    
    # Check for pipeline configuration
    if 'pipeline' in config:
        method_config = {'pipeline': config['pipeline']}
        return create_method_from_config(method_config)
    
    # Single method configuration
    if 'method' in config:
        method_config = {
            'method': config['method'],
            'params': config.get('params', {})
        }
        return create_method_from_config(method_config)
    
    raise ValueError("Invalid analysis configuration. Must specify 'method', 'pipeline', or 'method_config'")


def get_chunk_generator(chunks_dir: Path, chunk_pattern: str):
    """
    Generator for iterating through feature chunks.
    
    Args:
        chunks_dir: Directory containing chunk files
        chunk_pattern: Pattern for chunk filenames (e.g., "chunk_{:03d}.npz")
        
    Yields:
        Feature chunks as numpy arrays
    """
    chunk_files = sorted(chunks_dir.glob(chunk_pattern.replace("{:03d}", "*")))
    
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file, allow_pickle=True)
        yield chunk_data['features']


def get_coordinate_generator(chunks_dir: Path, chunk_pattern: str):
    """
    Generator for iterating through coordinate chunks.
    
    Args:
        chunks_dir: Directory containing chunk files
        chunk_pattern: Pattern for chunk filenames
        
    Yields:
        Tuples of (features, coordinates) from chunks
    """
    chunk_files = sorted(chunks_dir.glob(chunk_pattern.replace("{:03d}", "*")))
    
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file, allow_pickle=True)
        features = chunk_data['features']
        coordinates = chunk_data['coordinates'].tolist() if chunk_data['coordinates'].dtype == object else chunk_data['coordinates']
        yield features, coordinates