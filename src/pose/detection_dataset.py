"""
Dataset class for loading detection data in batches for pose estimation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import numpy as np


class DetectionDataset:
    """
    Dataset for loading detection data in batches.
    
    Handles memory-efficient loading of detection results with patch coordinates,
    cluster assignments, and depth information.
    """
    
    def __init__(self, detections_dir: Path):
        """
        Initialize dataset from directory containing detection pickle files.
        
        Args:
            detections_dir: Directory containing .pkl files with detection results
        """
        self.detections_dir = Path(detections_dir)
        
        if not self.detections_dir.exists():
            raise FileNotFoundError(f"Detections directory not found: {detections_dir}")
        
        # Cache file for indexing
        self.index_cache_file = self.detections_dir / ".detection_index.pkl"
        
        # Index all detection files (with caching)
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load cached index or create new one."""
        # Get all detection files (exclude cache and dataset_index files)
        self.detection_files = list(self.detections_dir.glob("*.pkl"))
        excluded_names = {".detection_index.pkl", "dataset_index.pkl"}
        self.detection_files = [f for f in self.detection_files if f.name not in excluded_names]
        
        if not self.detection_files:
            raise ValueError(f"No .pkl files found in {self.detections_dir}")
        
        # Check if cache exists and is newer than detection files
        if self._is_cache_valid():
            print("Loading cached detection index...")
            with open(self.index_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.detection_index = [(Path(f), idx) for f, idx in cache_data['detection_index']]
            self.total_detections = cache_data['total_detections']
            cached_files = cache_data['detection_files']
            
            # Verify cached files match current files
            if set(str(f) for f in self.detection_files) == set(cached_files):
                return
            else:
                print("Cache outdated, reindexing...")
        
        # Create new index
        print("Creating detection index...")
        self._index_detections()
        self._save_index()
    
    def _is_cache_valid(self) -> bool:
        """Check if cached index exists and is newer than detection files."""
        if not self.index_cache_file.exists():
            return False
        
        cache_time = self.index_cache_file.stat().st_mtime
        
        # Check if any detection file is newer than cache
        for file_path in self.detection_files:
            if file_path.stat().st_mtime > cache_time:
                return False
        
        return True
    
    def _index_detections(self) -> None:
        """Create index mapping global detection index to (file_path, local_index)."""
        # Following CLAUDE_STYLE_GUIDE.md Section 12: fail-fast, no error masking
        self.detection_index = []  # List of (file_path, local_index) tuples
        self.total_detections = 0
        
        for file_path in self.detection_files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Following CLAUDE_STYLE_GUIDE.md Section 12: no fallbacks that hide problems
            detailed_detections = data['detailed_detections']
            
            # Map each detection to its location
            for local_idx in range(len(detailed_detections)):
                self.detection_index.append((file_path, local_idx))
                self.total_detections += 1
    
    def _save_index(self) -> None:
        """Save index to cache file."""
        cache_data = {
            'detection_files': [str(f) for f in self.detection_files],
            'detection_index': [(str(f), idx) for f, idx in self.detection_index],
            'total_detections': self.total_detections
        }
        
        with open(self.index_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cached index with {self.total_detections} detections")
    
    def __len__(self) -> int:
        """Return total number of detections across all files."""
        return self.total_detections
    
    def _group_indices_by_file(self, start_idx: int, batch_size: int) -> Dict[Path, List[int]]:
        """
        Group detection indices by file for efficient loading.
        
        Following CLAUDE_STYLE_GUIDE.md Section 5 (lines 32-35): "Create small, focused helper functions"
        """
        file_groups = {}
        
        for global_idx in range(start_idx, min(start_idx + batch_size, self.total_detections)):
            file_path, local_idx = self.detection_index[global_idx]
            
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(local_idx)
        
        return file_groups
    
    def load_detection_batch(self, start_idx: int, batch_size: int) -> List[Dict]:
        """
        Load a batch of detections starting from start_idx.
        
        Following CLAUDE_STYLE_GUIDE.md Section 6 (lines 37-40): "Functions should be independently usable"
        
        Args:
            start_idx: Starting detection index
            batch_size: Number of detections to load
            
        Returns:
            List of detection dictionaries with patch data
        """
        # Following CLAUDE_STYLE_GUIDE.md Section 5 (lines 32-35): "Main functions should orchestrate"
        file_groups = self._group_indices_by_file(start_idx, batch_size)
        detections = []
        
        # Load each file once and extract needed detections
        for file_path, local_indices in file_groups.items():
            # Following CLAUDE_STYLE_GUIDE.md Section 12: fail-fast, no error masking
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            detailed_detections = data['detailed_detections']
            
            # Extract needed detections from this file
            for local_idx in local_indices:
                detections.append(detailed_detections[local_idx])
        
        return detections
    
    
    def iter_batches(self, batch_size: int) -> Iterator[List[Dict]]:
        """
        Iterate over all detections in batches.
        
        Args:
            batch_size: Number of detections per batch
            
        Yields:
            List of detection dictionaries
        """
        start_idx = 0
        
        while start_idx < self.total_detections:
            batch = self.load_detection_batch(start_idx, batch_size)
            
            if not batch:
                break
            
            yield batch
            start_idx += len(batch)
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_files': len(self.detection_files),
            'total_detections': self.total_detections,
            'avg_detections_per_file': self.total_detections / len(self.detection_files) if self.detection_files else 0
        }