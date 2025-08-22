"""Modular patch analysis system supporting multiple methods (PCA, clustering, combinations)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class PatchAnalysisResult:
    """Container for patch analysis results from any method."""
    
    def __init__(
        self,
        method_name: str,
        patch_features: np.ndarray,
        patch_coordinates: np.ndarray,
        method_output: Any,
        n_components: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize analysis result.
        
        Args:
            method_name: Name of the analysis method used
            patch_features: Original patch features (n_patches, feature_dim)
            patch_coordinates: Patch coordinates (n_patches, 2)
            method_output: Method-specific output (components, labels, probabilities, etc.)
            n_components: Number of components/clusters discovered
            metadata: Additional method-specific information
        """
        self.method_name = method_name
        self.patch_features = patch_features
        self.patch_coordinates = patch_coordinates
        self.method_output = method_output
        self.n_components = n_components
        self.metadata = metadata or {}
    
    def get_visualization_data(self) -> np.ndarray:
        """
        Get data suitable for visualization (n_patches, n_components).
        
        Returns:
            Array where each row represents a patch and columns represent 
            component weights, cluster probabilities, or transformed features.
        """
        if isinstance(self.method_output, np.ndarray):
            if self.method_output.ndim == 1:
                # Cluster labels - convert to one-hot encoding
                unique_labels = np.unique(self.method_output)
                n_clusters = len(unique_labels) 
                one_hot = np.zeros((len(self.method_output), n_clusters))
                
                # Map labels to indices (in case labels aren't 0,1,2,...)
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                indices = np.array([label_to_idx[label] for label in self.method_output])
                one_hot[np.arange(len(self.method_output)), indices] = 1.0
                return one_hot
            else:
                # Component scores or probabilities
                return self.method_output
        else:
            raise ValueError(f"Unsupported method output type: {type(self.method_output)}")


class PatchAnalysisMethod(ABC):
    """Abstract base class for patch analysis methods."""
    
    @abstractmethod
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit the method to patch features."""
        pass
    
    @abstractmethod
    def transform(self, patch_features: np.ndarray) -> Any:
        """Transform patch features using fitted method."""
        pass
    
    @abstractmethod
    def fit_transform(self, patch_features: np.ndarray) -> Any:
        """Fit and transform in one step."""
        pass
    
    @abstractmethod
    def get_n_components(self) -> int:
        """Get number of components/clusters."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get human-readable method name."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get method-specific metadata."""
        pass
    
    @abstractmethod
    def supports_incremental_fit(self) -> bool:
        """Whether this method supports incremental/online fitting."""
        pass
    
    def incremental_fit(self, patch_features: np.ndarray) -> None:
        """Incremental fit (optional, override if supported)."""
        if self.supports_incremental_fit():
            raise NotImplementedError("Method claims incremental support but doesn't implement incremental_fit")
        else:
            raise NotImplementedError("Method doesn't support incremental fitting")


def analyze_patches(
    method: PatchAnalysisMethod,
    patch_features: np.ndarray,
    patch_coordinates: np.ndarray,
    fit_method: bool = True
) -> PatchAnalysisResult:
    """
    Analyze patches using specified method.
    
    Args:
        method: Analysis method instance
        patch_features: Patch features (n_patches, feature_dim)
        patch_coordinates: Patch coordinates (n_patches, 2)
        fit_method: Whether to fit the method (False for pre-fitted methods)
        
    Returns:
        PatchAnalysisResult containing all analysis outputs
    """
    if fit_method:
        method_output = method.fit_transform(patch_features)
    else:
        method_output = method.transform(patch_features)
    
    return PatchAnalysisResult(
        method_name=method.get_method_name(),
        patch_features=patch_features,
        patch_coordinates=patch_coordinates,
        method_output=method_output,
        n_components=method.get_n_components(),
        metadata=method.get_metadata()
    )