"""Implementations of different patch analysis methods."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.mixture import GaussianMixture

from .patch_analyzer import PatchAnalysisMethod


class IncrementalPCAMethod(PatchAnalysisMethod):
    """PCA-based patch analysis using IncrementalPCA."""
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = IncrementalPCA(n_components=n_components)
        self._is_fitted = False
    
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit PCA to patch features."""
        self.pca.fit(patch_features)
        self._is_fitted = True
    
    def transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Transform patch features to PCA space."""
        if not self._is_fitted:
            raise ValueError("Method must be fitted before transform")
        return self.pca.transform(patch_features)
    
    def fit_transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Fit PCA and transform features."""
        result = self.pca.fit_transform(patch_features)
        self._is_fitted = True
        return result
    
    def incremental_fit(self, patch_features: np.ndarray) -> None:
        """Incremental fitting for large datasets."""
        self.pca.partial_fit(patch_features)
        self._is_fitted = True
    
    def get_n_components(self) -> int:
        return self.n_components
    
    def get_method_name(self) -> str:
        return f"Incremental PCA ({self.n_components} components)"
    
    def get_metadata(self) -> Dict[str, Any]:
        if self._is_fitted:
            return {
                'explained_variance_ratio': self.pca.explained_variance_ratio_,
                'total_explained_variance': self.pca.explained_variance_ratio_.sum(),
                'n_components': self.pca.n_components_
            }
        return {}
    
    def supports_incremental_fit(self) -> bool:
        return True


class MiniBatchKMeansMethod(PatchAnalysisMethod):
    """K-Means clustering using MiniBatchKMeans."""
    
    def __init__(self, n_clusters: int = 10, batch_size: int = 1000, random_state: int = 42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=batch_size,
            random_state=random_state
        )
        self._is_fitted = False
    
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit K-Means to patch features."""
        self.kmeans.fit(patch_features)
        self._is_fitted = True
    
    def transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Get cluster labels for patch features."""
        if not self._is_fitted:
            raise ValueError("Method must be fitted before transform")
        return self.kmeans.predict(patch_features)
    
    def fit_transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Fit K-Means and get cluster labels."""
        labels = self.kmeans.fit_predict(patch_features)
        self._is_fitted = True
        return labels
    
    def incremental_fit(self, patch_features: np.ndarray) -> None:
        """Incremental fitting for large datasets."""
        self.kmeans.partial_fit(patch_features)
        self._is_fitted = True
    
    def get_n_components(self) -> int:
        return self.n_clusters
    
    def get_method_name(self) -> str:
        return f"MiniBatch K-Means ({self.n_clusters} clusters)"
    
    def get_metadata(self) -> Dict[str, Any]:
        if self._is_fitted:
            return {
                'cluster_centers': self.kmeans.cluster_centers_,
                'inertia': self.kmeans.inertia_,
                'n_iter': getattr(self.kmeans, 'n_iter_', None)
            }
        return {}
    
    def supports_incremental_fit(self) -> bool:
        return True


class BirchMethod(PatchAnalysisMethod):
    """BIRCH clustering method."""
    
    def __init__(self, n_clusters: int = 10, threshold: float = 0.5):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.birch = Birch(n_clusters=n_clusters, threshold=threshold)
        self._is_fitted = False
    
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit BIRCH to patch features."""
        self.birch.fit(patch_features)
        self._is_fitted = True
    
    def transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Get cluster labels for patch features."""
        if not self._is_fitted:
            raise ValueError("Method must be fitted before transform")
        return self.birch.predict(patch_features)
    
    def fit_transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Fit BIRCH and get cluster labels."""
        labels = self.birch.fit_predict(patch_features)
        self._is_fitted = True
        return labels
    
    def incremental_fit(self, patch_features: np.ndarray) -> None:
        """Incremental fitting for large datasets."""
        self.birch.partial_fit(patch_features)
        self._is_fitted = True
    
    def get_n_components(self) -> int:
        return self.n_clusters
    
    def get_method_name(self) -> str:
        return f"BIRCH ({self.n_clusters} clusters)"
    
    def get_metadata(self) -> Dict[str, Any]:
        if self._is_fitted:
            return {
                'n_features_in': getattr(self.birch, 'n_features_in_', None),
                'threshold': self.threshold
            }
        return {}
    
    def supports_incremental_fit(self) -> bool:
        return True


class GaussianMixtureMethod(PatchAnalysisMethod):
    """Gaussian Mixture Model for soft clustering."""
    
    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self._is_fitted = False
    
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit GMM to patch features."""
        self.gmm.fit(patch_features)
        self._is_fitted = True
    
    def transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Get cluster probabilities for patch features."""
        if not self._is_fitted:
            raise ValueError("Method must be fitted before transform")
        return self.gmm.predict_proba(patch_features)
    
    def fit_transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Fit GMM and get cluster probabilities."""
        self.gmm.fit(patch_features)
        self._is_fitted = True
        return self.gmm.predict_proba(patch_features)
    
    def get_n_components(self) -> int:
        return self.n_components
    
    def get_method_name(self) -> str:
        return f"Gaussian Mixture ({self.n_components} components)"
    
    def get_metadata(self) -> Dict[str, Any]:
        if self._is_fitted:
            return {
                'means': self.gmm.means_,
                'covariances': self.gmm.covariances_,
                'weights': self.gmm.weights_,
                'aic': self.gmm.aic(self.gmm.means_) if hasattr(self.gmm, 'aic') else None,
                'bic': self.gmm.bic(self.gmm.means_) if hasattr(self.gmm, 'bic') else None
            }
        return {}
    
    def supports_incremental_fit(self) -> bool:
        return False  # Standard GMM doesn't support incremental fitting


class PipelineMethod(PatchAnalysisMethod):
    """Pipeline combining multiple methods (e.g., PCA → K-Means)."""
    
    def __init__(self, methods: list[PatchAnalysisMethod], method_names: Optional[list[str]] = None):
        """
        Initialize pipeline with multiple methods.
        
        Args:
            methods: List of methods to apply in sequence
            method_names: Optional custom names for methods
        """
        self.methods = methods
        self.method_names = method_names or [m.get_method_name() for m in methods]
        self._is_fitted = False
        self._first_method_fitted = False
    
    def fit(self, patch_features: np.ndarray) -> None:
        """Fit all methods in sequence."""
        current_features = patch_features
        for method in self.methods:
            method.fit(current_features)
            current_features = method.transform(current_features)
        self._is_fitted = True
        self._first_method_fitted = True
    
    def transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Transform features through all methods."""
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        current_features = patch_features
        for method in self.methods:
            current_features = method.transform(current_features)
        return current_features
    
    def fit_transform(self, patch_features: np.ndarray) -> np.ndarray:
        """Fit pipeline and transform features."""
        current_features = patch_features
        for method in self.methods:
            current_features = method.fit_transform(current_features)
        self._is_fitted = True
        self._first_method_fitted = True
        return current_features
    
    def incremental_fit(self, patch_features: np.ndarray) -> None:
        """Phase 1: Only fit first method incrementally."""
        if not self.methods:
            return
        
        first_method = self.methods[0]
        if not first_method.supports_incremental_fit():
            raise ValueError("First method in pipeline must support incremental fitting")
        
        first_method.incremental_fit(patch_features)
        self._first_method_fitted = True
    
    def finalize_pipeline(self, feature_generator) -> None:
        """
        Phase 2: Complete pipeline by fitting remaining methods.
        
        Args:
            feature_generator: Generator that yields feature chunks
        """
        if len(self.methods) <= 1:
            self._is_fitted = True
            return
        
        if not self._first_method_fitted:
            raise ValueError("Must call incremental_fit before finalize_pipeline")
        
        # Collect all transformed features from first method (now final)
        all_transformed_chunks = []
        for chunk_features in feature_generator:
            transformed_chunk = self.methods[0].transform(chunk_features)
            all_transformed_chunks.append(transformed_chunk)
        
        # Concatenate for subsequent method fitting
        all_transformed = np.vstack(all_transformed_chunks)
        
        # Fit remaining methods in sequence using fit()
        current_features = all_transformed
        for method in self.methods[1:]:
            method.fit(current_features)
            current_features = method.transform(current_features)
        
        self._is_fitted = True
    
    def get_n_components(self) -> int:
        """Get number of components from final method."""
        return self.methods[-1].get_n_components()
    
    def get_method_name(self) -> str:
        return " → ".join(self.method_names)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from all methods."""
        metadata = {}
        for i, method in enumerate(self.methods):
            method_meta = method.get_metadata()
            if method_meta:
                metadata[f"method_{i}_{self.method_names[i]}"] = method_meta
        return metadata
    
    def supports_incremental_fit(self) -> bool:
        """Pipeline supports incremental fit only if first method does."""
        return len(self.methods) > 0 and self.methods[0].supports_incremental_fit()