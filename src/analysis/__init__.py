"""Modular patch analysis system."""

from .method_registry import (
    COMMON_CONFIGS,
    MethodRegistry,
    create_common_method,
    create_method_from_config,
    get_method_info,
)
from .methods import (
    BirchMethod,
    GaussianMixtureMethod,
    IncrementalPCAMethod,
    MiniBatchKMeansMethod,
    PipelineMethod,
)
from .patch_analyzer import PatchAnalysisMethod, PatchAnalysisResult, analyze_patches

__all__ = [
    # Core classes
    'PatchAnalysisMethod',
    'PatchAnalysisResult',
    'analyze_patches',
    
    # Method implementations
    'IncrementalPCAMethod',
    'MiniBatchKMeansMethod',
    'BirchMethod', 
    'GaussianMixtureMethod',
    'PipelineMethod',
    
    # Registry system
    'MethodRegistry',
    'create_method_from_config',
    'create_common_method',
    'get_method_info',
    'COMMON_CONFIGS',
]