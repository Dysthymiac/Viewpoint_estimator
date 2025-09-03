"""CVAT annotation loader for viewpoint analysis."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def crop_image(self, image: Image.Image) -> Image.Image:
        """Crop image using bounding box coordinates."""
        return image.crop((self.x1, self.y1, self.x2, self.y2))
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class Annotation:
    """Single image annotation with viewpoint label."""
    image_path: Path
    bbox: BoundingBox
    viewpoint: str
    image_id: int


class CVATLoader:
    """Loads CVAT annotations and provides image access."""
    
    def __init__(self, dataset_root: Path, crop_to_bbox: bool = True) -> None:
        self.dataset_root = Path(dataset_root)
        self.annotations_path = self.dataset_root / "annotations.xml"
        self.images_dir = self.dataset_root / "images"
        self.crop_to_bbox = crop_to_bbox
        self._annotations: list[Annotation] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parse CVAT XML annotations."""
        tree = ET.parse(self.annotations_path)
        root = tree.getroot()
        
        for image_elem in root.findall('image'):
            image_id = int(image_elem.get('id'))
            image_name = image_elem.get('name')
            image_path = self.images_dir / image_name
            
            for box_elem in image_elem.findall('box'):
                bbox = BoundingBox(
                    x1=float(box_elem.get('xtl')),
                    y1=float(box_elem.get('ytl')),
                    x2=float(box_elem.get('xbr')),
                    y2=float(box_elem.get('ybr'))
                )
                
                viewpoint_elem = box_elem.find('.//attribute[@name="Viewpoint"]')
                viewpoint = viewpoint_elem.text if viewpoint_elem is not None else "Unknown"
                
                self._annotations.append(Annotation(
                    image_path=image_path,
                    bbox=bbox,
                    viewpoint=viewpoint,
                    image_id=image_id
                ))

    @property
    def annotations(self) -> list[Annotation]:
        """Get all annotations."""
        return self._annotations.copy()

    @property
    def viewpoints(self) -> set[str]:
        """Get unique viewpoint labels."""
        return {ann.viewpoint for ann in self._annotations}

    def load_cropped_image(self, annotation: Annotation) -> Image.Image:
        """Load and crop image according to annotation."""
        image = Image.open(annotation.image_path).convert('RGB')
        return annotation.bbox.crop_image(image)
    
    def load_full_image(self, annotation: Annotation) -> Image.Image:
        """Load full image without cropping."""
        return Image.open(annotation.image_path).convert('RGB')
    
    def load_image(self, annotation: Annotation) -> Image.Image:
        """Load image according to crop_to_bbox setting."""
        if self.crop_to_bbox:
            return self.load_cropped_image(annotation)
        else:
            return self.load_full_image(annotation)

    def iter_images(self) -> Iterator[tuple[Annotation, Image.Image]]:
        """Iterate over annotations with cropped images."""
        for annotation in self._annotations:
            yield annotation, self.load_cropped_image(annotation)

    def __len__(self) -> int:
        return len(self._annotations)