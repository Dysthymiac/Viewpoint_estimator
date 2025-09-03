"""Simple folder-based image loader for viewpoint analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .cvat_loader import BoundingBox


@dataclass(frozen=True)
class FolderAnnotation:
    """Single image annotation for folder-based loader."""
    image_path: Path
    image_id: str
    
    @property
    def bbox(self) -> BoundingBox:
        """Return full image bbox (no cropping for folder loader)."""
        # Load image to get dimensions
        with Image.open(self.image_path) as img:
            width, height = img.size
        return BoundingBox(0, 0, width, height)
    
    @property
    def viewpoint(self) -> str:
        """Default viewpoint for folder loader."""
        return "unknown"


class FolderLoader:
    """Loads images from a simple folder structure."""
    
    def __init__(self, folder_path: Path, crop_to_bbox: bool = False, image_extensions: list[str] | None = None) -> None:
        self.folder_path = Path(folder_path)
        self.crop_to_bbox = crop_to_bbox
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self._annotations: list[FolderAnnotation] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Scan folder for image files."""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder path does not exist: {self.folder_path}")
        
        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.folder_path}")
        
        # Find all image files recursively
        for image_path in self.folder_path.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                # Use relative path from folder root as image_id
                relative_path = image_path.relative_to(self.folder_path)
                image_id = str(relative_path)
                
                self._annotations.append(FolderAnnotation(
                    image_path=image_path,
                    image_id=image_id
                ))

    @property
    def annotations(self) -> list[FolderAnnotation]:
        """Get all annotations."""
        return self._annotations.copy()

    @property
    def viewpoints(self) -> set[str]:
        """Get unique viewpoint labels (always 'unknown' for folder loader)."""
        return {"unknown"}

    def load_cropped_image(self, annotation: FolderAnnotation) -> Image.Image:
        """Load and crop image according to annotation (no cropping for folder loader)."""
        return self.load_full_image(annotation)
    
    def load_full_image(self, annotation: FolderAnnotation) -> Image.Image:
        """Load full image without cropping."""
        return Image.open(annotation.image_path).convert('RGB')
    
    def load_image(self, annotation: FolderAnnotation) -> Image.Image:
        """Load image according to crop_to_bbox setting."""
        if self.crop_to_bbox:
            return self.load_cropped_image(annotation)
        else:
            return self.load_full_image(annotation)

    def iter_images(self) -> Iterator[tuple[FolderAnnotation, Image.Image]]:
        """Iterate over annotations with images."""
        for annotation in self._annotations:
            yield annotation, self.load_image(annotation)

    def __len__(self) -> int:
        return len(self._annotations)