"""
Cluster Utility Functions

Standalone utility functions for cluster operations used across the codebase.
Following style guide Lines 36-40: Reusable utility functions.
"""

from __future__ import annotations

from typing import List
import numpy as np


def combine_cluster_patches(patch_components: np.ndarray, cluster_ids: List[int]) -> np.ndarray:
    """
    Combine patches from multiple clusters into single mask.
    
    Style guide Lines 5-9: No duplication ✓ - Centralized utility function
    Style guide Lines 11-15: SRP ✓ - ONLY combines clusters
    Style guide Lines 36-40: Reusability ✓ - Used by both core algorithm and visualization
    
    Args:
        patch_components: Cluster membership matrix (n_patches, n_clusters)
        cluster_ids: List of cluster indices to combine
    
    Returns:
        Combined cluster mask (n_patches,) with values [0, 1]
    """
    cluster_mask = np.zeros(len(patch_components))
    for cluster_id in cluster_ids:
        if cluster_id < patch_components.shape[1]:
            cluster_mask += patch_components[:, cluster_id]
    return np.clip(cluster_mask, 0, 1)