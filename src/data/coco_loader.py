"""COCO format annotation loader for viewpoint analysis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .cvat_loader import BoundingBox


@dataclass(frozen=True)
class COCOImage:
    """COCO image metadata."""
    uuid: str
    file_name: str
    width: int
    height: int
    latitude: float
    longitude: float
    datetime: str


@dataclass(frozen=True)
class COCOAnnotation:
    """Single COCO annotation with viewpoint label."""
    annot_uuid: str
    image_uuid: str
    bbox: BoundingBox
    viewpoint: str
    category: str
    category_id: int
    manual: bool
    census_annot: bool
    
    @property
    def image_id(self) -> str:
        """Get image UUID for compatibility."""
        return self.image_uuid


@dataclass(frozen=True)
class COCOCategory:
    """COCO category information."""
    id: int
    name: str


class COCOLoader:
    """Loads COCO format annotations and provides image access."""
    
    def __init__(self, dataset_root: Path, coco_json_path: Path, crop_to_bbox: bool = True) -> None:
        self.dataset_root = Path(dataset_root)
        self.coco_json_path = Path(coco_json_path)
        self.crop_to_bbox = crop_to_bbox
        self._annotations: list[COCOAnnotation] = []
        self._images: dict[str, COCOImage] = {}
        self._categories: dict[int, COCOCategory] = {}
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Parse COCO JSON annotations."""
        with open(self.coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Load categories
        for cat_data in coco_data['categories']:
            category = COCOCategory(
                id=cat_data['id'],
                name=cat_data['name']
            )
            self._categories[category.id] = category
        
        # Load images
        for img_data in coco_data['images']:
            image = COCOImage(
                uuid=img_data['uuid'],
                file_name=img_data['file_name'],
                width=img_data['width'],
                height=img_data['height'],
                latitude=img_data['latitude'],
                longitude=img_data['longitude'],
                datetime=img_data['datetime']
            )
            self._images[image.uuid] = image
        
        # Load annotations
        for ann_data in coco_data['annotations']:
            bbox = BoundingBox(
                x1=float(ann_data['bbox x']),
                y1=float(ann_data['bbox y']),
                x2=float(ann_data['bbox x']) + float(ann_data['bbox w']),
                y2=float(ann_data['bbox y']) + float(ann_data['bbox h'])
            )
            
            # Ensure viewpoint is always a string
            viewpoint = ann_data['annot viewpoint']
            if not isinstance(viewpoint, str):
                viewpoint = "unknown"
            
            annotation = COCOAnnotation(
                annot_uuid=ann_data['annot uuid'],
                image_uuid=ann_data['image uuid'],
                bbox=bbox,
                viewpoint=viewpoint,
                category=ann_data['category'],
                category_id=ann_data['category_id'],
                manual=ann_data['annot manual'],
                census_annot=ann_data['census annot']
            )
            
            self._annotations.append(annotation)

    @property
    def annotations(self) -> list[COCOAnnotation]:
        """Get all annotations."""
        return self._annotations.copy()

    @property
    def images(self) -> dict[str, COCOImage]:
        """Get all images indexed by UUID."""
        return self._images.copy()

    @property
    def categories(self) -> dict[int, COCOCategory]:
        """Get all categories indexed by ID."""
        return self._categories.copy()

    @property
    def viewpoints(self) -> set[str]:
        """Get unique viewpoint labels."""
        return {ann.viewpoint for ann in self._annotations}

    def get_image_path(self, annotation: COCOAnnotation) -> Path:
        """Get full path to image file."""
        image = self._images[annotation.image_uuid]
        return self.dataset_root / image.file_name

    def load_cropped_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load and crop image according to annotation."""
        image_path = self.get_image_path(annotation)
        image = Image.open(image_path).convert('RGB')
        return annotation.bbox.crop_image(image)
    
    def load_full_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load full image without cropping."""
        image_path = self.get_image_path(annotation)
        return Image.open(image_path).convert('RGB')
    
    def load_image(self, annotation: COCOAnnotation) -> Image.Image:
        """Load image according to crop_to_bbox setting."""
        if self.crop_to_bbox:
            return self.load_cropped_image(annotation)
        else:
            return self.load_full_image(annotation)

    def filter_by_category(self, category_names: list[str]) -> list[COCOAnnotation]:
        """Filter annotations by category names."""
        return [ann for ann in self._annotations if ann.category in category_names]

    def filter_by_viewpoint(self, viewpoints: list[str]) -> list[COCOAnnotation]:
        """Filter annotations by viewpoint labels."""
        return [ann for ann in self._annotations if ann.viewpoint in viewpoints]

    def filter_manual_annotations(self) -> list[COCOAnnotation]:
        """Get only manually annotated items."""
        return [ann for ann in self._annotations if ann.manual]

    def iter_images(self) -> Iterator[tuple[COCOAnnotation, Image.Image]]:
        """Iterate over annotations with images."""
        for annotation in self._annotations:
            yield annotation, self.load_image(annotation)

    def __len__(self) -> int:
        return len(self._annotations)