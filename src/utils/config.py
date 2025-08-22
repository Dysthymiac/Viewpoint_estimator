"""Configuration loading and directory setup utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_directories(config: dict[str, Any]) -> None:
    """Create output directories specified in config."""
    directories_to_create = [
        config['features']['output_dir'],
        config['visualization']['output_dir'],
        config['logging']['log_dir']
    ]
    
    # Add analysis output directory (new modular system)
    if 'analysis' in config:
        directories_to_create.append(config['analysis']['output_dir'])
    
    # Add legacy PCA directory if present (backward compatibility)
    if 'pca' in config:
        directories_to_create.append(config['pca']['output_dir'])
    
    for dir_path in directories_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)