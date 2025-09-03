"""
Optimized dataset class for loading consolidated batch detection files.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Iterator
import numpy as np


class BatchDetectionDataset:
    """
    Dataset for loading consolidated batch detection files.
    
    Much faster I/O than loading individual files.
    """
    
    def __init__(self, batch_detections_dir: Path):
        """
        Initialize dataset from directory containing batch pickle files.
        
        Args:
            batch_detections_dir: Directory containing batch_*.pkl files
        """
        self.batch_detections_dir = Path(batch_detections_dir)
        
        if not self.batch_detections_dir.exists():
            raise FileNotFoundError(f"Batch detections directory not found: {batch_detections_dir}")
        
        # Get all batch files
        self.batch_files = sorted(list(self.batch_detections_dir.glob("batch_*.pkl")))
        
        if not self.batch_files:
            raise ValueError(f"No batch_*.pkl files found in {batch_detections_dir}")
        
        # Count total detections by reading batch headers
        self._count_total_detections()
    
    def _count_total_detections(self) -> None:
        """Count total detections across all batch files."""
        self.total_detections = 0
        
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
            self.total_detections += data['num_detections']
    
    def iter_batches(self, batch_size: int) -> Iterator[List[Dict]]:
        """
        Iterate over detections in batches.
        
        Args:
            batch_size: Number of detections per yielded batch
            
        Yields:
            Lists of detection dictionaries
        """
        current_batch = []
        
        for batch_file in self.batch_files:
            # Load entire batch file at once (much faster than individual files)
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
            
            detections = data['detailed_detections']
            
            for detection in detections:
                current_batch.append(detection)
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
        
        # Yield remaining detections
        if current_batch:
            yield current_batch
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'total_detections': self.total_detections,
            'num_batch_files': len(self.batch_files),
            'avg_detections_per_file': self.total_detections / len(self.batch_files) if self.batch_files else 0
        }