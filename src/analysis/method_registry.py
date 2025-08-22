"""Registry system for easy method selection and configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .methods import (
    BirchMethod,
    GaussianMixtureMethod,
    IncrementalPCAMethod,
    MiniBatchKMeansMethod,
    PipelineMethod,
)
from .patch_analyzer import PatchAnalysisMethod


class MethodRegistry:
    """Registry for managing and creating patch analysis methods."""
    
    _methods: Dict[str, Type[PatchAnalysisMethod]] = {
        'pca': IncrementalPCAMethod,
        'kmeans': MiniBatchKMeansMethod,
        'birch': BirchMethod,
        'gmm': GaussianMixtureMethod,
    }
    
    @classmethod
    def register_method(cls, name: str, method_class: Type[PatchAnalysisMethod]) -> None:
        """Register a new method class."""
        cls._methods[name] = method_class
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available method names."""
        return list(cls._methods.keys())
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> PatchAnalysisMethod:
        """
        Create method instance by name.
        
        Args:
            method_name: Name of the method to create
            **kwargs: Method-specific parameters
            
        Returns:
            Method instance
        """
        if method_name not in cls._methods:
            available = ', '.join(cls.get_available_methods())
            raise ValueError(f"Unknown method '{method_name}'. Available: {available}")
        
        method_class = cls._methods[method_name]
        return method_class(**kwargs)
    
    @classmethod
    def create_pipeline(cls, method_configs: List[Dict[str, Any]]) -> PipelineMethod:
        """
        Create pipeline from list of method configurations.
        
        Args:
            method_configs: List of dicts with 'name' and optional 'params'
            
        Example:
            create_pipeline([
                {'name': 'pca', 'params': {'n_components': 50}},
                {'name': 'kmeans', 'params': {'n_clusters': 10}}
            ])
        """
        methods = []
        method_names = []
        
        for config in method_configs:
            method_name = config['name']
            params = config.get('params', {})
            
            method = cls.create_method(method_name, **params)
            methods.append(method)
            method_names.append(method.get_method_name())
        
        return PipelineMethod(methods, method_names)


def create_method_from_config(config: Dict[str, Any]) -> PatchAnalysisMethod:
    """
    Create method from configuration dictionary.
    
    Args:
        config: Configuration with 'method' or 'pipeline' key
        
    Examples:
        # Single method
        config = {
            'method': 'pca',
            'params': {'n_components': 50}
        }
        
        # Pipeline
        config = {
            'pipeline': [
                {'name': 'pca', 'params': {'n_components': 50}},
                {'name': 'kmeans', 'params': {'n_clusters': 10}}
            ]
        }
    """
    if 'pipeline' in config:
        return MethodRegistry.create_pipeline(config['pipeline'])
    elif 'method' in config:
        method_name = config['method']
        params = config.get('params', {})
        return MethodRegistry.create_method(method_name, **params)
    else:
        raise ValueError("Config must contain either 'method' or 'pipeline' key")


# Predefined common configurations
COMMON_CONFIGS = {
    'pca_50': {
        'method': 'pca',
        'params': {'n_components': 50}
    },
    'pca_100': {
        'method': 'pca', 
        'params': {'n_components': 100}
    },
    'kmeans_10': {
        'method': 'kmeans',
        'params': {'n_clusters': 10}
    },
    'kmeans_20': {
        'method': 'kmeans',
        'params': {'n_clusters': 20}
    },
    'gmm_10': {
        'method': 'gmm',
        'params': {'n_components': 10}
    },
    'birch_10': {
        'method': 'birch',
        'params': {'n_clusters': 10}
    },
    'pca_kmeans': {
        'pipeline': [
            {'name': 'pca', 'params': {'n_components': 50}},
            {'name': 'kmeans', 'params': {'n_clusters': 10}}
        ]
    },
    'pca_gmm': {
        'pipeline': [
            {'name': 'pca', 'params': {'n_components': 50}},
            {'name': 'gmm', 'params': {'n_components': 10}}
        ]
    }
}


def create_common_method(config_name: str) -> PatchAnalysisMethod:
    """Create method from predefined common configuration."""
    if config_name not in COMMON_CONFIGS:
        available = ', '.join(COMMON_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return create_method_from_config(COMMON_CONFIGS[config_name])


def get_method_info(method_name: str) -> Dict[str, Any]:
    """Get information about a method."""
    if method_name not in MethodRegistry._methods:
        return {}
    
    method_class = MethodRegistry._methods[method_name]
    
    # Get default instance to extract info
    try:
        if method_name == 'pca':
            instance = method_class(n_components=10)
        elif method_name in ['kmeans', 'birch']:
            instance = method_class(n_clusters=10)
        elif method_name == 'gmm':
            instance = method_class(n_components=10)
        else:
            instance = method_class()
        
        return {
            'name': method_name,
            'class': method_class.__name__,
            'supports_incremental': instance.supports_incremental_fit(),
            'method_name': instance.get_method_name()
        }
    except Exception:
        return {
            'name': method_name,
            'class': method_class.__name__,
            'supports_incremental': 'unknown',
            'method_name': 'unknown'
        }