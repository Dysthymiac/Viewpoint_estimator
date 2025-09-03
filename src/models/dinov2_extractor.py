"""DINOv2 patch feature extractor with smart resize and aspect ratio preservation."""

from __future__ import annotations

import math
from typing import Tuple, Optional
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def smart_resize_with_padding(image: Image.Image, target_size: int) -> tuple[Image.Image, dict]:
    """
    Resize image preserving aspect ratio with padding to square.
    
    Args:
        image: PIL Image to resize
        target_size: Target square size
        
    Returns:
        Tuple of (resized_padded_image, padding_info)
        padding_info contains: {'top': int, 'bottom': int, 'left': int, 'right': int, 'scale': float}
    """
    original_width, original_height = image.size
    
    # Determine which dimension is limiting and calculate exact dimensions
    if original_width >= original_height:
        # Width is limiting dimension
        new_width = target_size
        new_height = int(original_height * target_size / original_width)
    else:
        # Height is limiting dimension  
        new_height = target_size
        new_width = int(original_width * target_size / original_height)
    
    # Calculate actual scale used
    scale = new_width / original_width
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate padding
    pad_width = target_size - new_width
    pad_height = target_size - new_height
    
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    
    # Create padded image (pad with black/zero)
    padded_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    padded_image.paste(resized_image, (pad_left, pad_top))
    
    padding_info = {
        'top': pad_top,
        'bottom': pad_bottom,
        'left': pad_left,
        'right': pad_right,
        'scale': scale
    }
    
    return padded_image, padding_info


def filter_padded_patches(patch_coords: np.ndarray, padding_info: dict, patch_size: int, image_size: int) -> np.ndarray:
    """
    Filter out patch coordinates that contain padded areas.
    
    Args:
        patch_coords: Array of shape (n_patches, 2) with (row, col) coordinates
        padding_info: Padding information from smart_resize_with_padding
        patch_size: Size of each patch in pixels
        image_size: Total image size
        
    Returns:
        Boolean mask of valid patches (True = keep, False = discard)
    """
    # Convert patch coordinates to pixel coordinates (top-left corner of each patch)
    patch_pixel_row = patch_coords[:, 0] * patch_size
    patch_pixel_col = patch_coords[:, 1] * patch_size
    
    # Define valid region boundaries (excluding padding)
    valid_left = padding_info['left']
    valid_right = image_size - padding_info['right']
    valid_top = padding_info['top']
    valid_bottom = image_size - padding_info['bottom']
    
    # Calculate patch boundaries
    patch_left = patch_pixel_col
    patch_right = patch_pixel_col + patch_size
    patch_top = patch_pixel_row
    patch_bottom = patch_pixel_row + patch_size
    
    # Patch is valid if it's completely within valid region
    valid_mask = (
        (patch_left >= valid_left) &
        (patch_right <= valid_right) &
        (patch_top >= valid_top) &
        (patch_bottom <= valid_bottom)
    )
    
    return valid_mask


def convert_to_relative_coordinates(
    patch_coords: np.ndarray,
    padding_info: dict,
    original_image_size: tuple[int, int],
    patch_size: int
) -> tuple[np.ndarray, float]:
    """
    Convert DINOv2 grid coordinates to relative coordinates [0,1] in original image space.
    
    Args:
        patch_coords: Shape (n_patches, 2) with (row, col) grid coordinates
        padding_info: Padding information from smart_resize_with_padding
        original_image_size: (width, height) of original image
        patch_size: DINOv2 patch size in pixels
        
    Returns:
        Tuple of (relative_coordinates, relative_patch_size)
        - relative_coordinates: Shape (n_patches, 2) with values in [0,1]
        - relative_patch_size: Float representing patch size relative to image dimensions
    """
    # Convert DINOv2 grid coords to pixel coords in padded space
    pixel_coords = patch_coords * patch_size
    
    # Remove padding to get coords in resized space
    padding_offset = np.array([padding_info['top'], padding_info['left']])
    resized_coords = pixel_coords - padding_offset
    
    # Convert back to original image pixel coordinates
    scale = padding_info['scale']
    original_pixel_coords = resized_coords / scale
    
    # Calculate actual patch size in original image space
    actual_patch_size_pixels = patch_size / scale
    
    # Normalize to relative coordinates [0,1]
    original_width, original_height = original_image_size
    relative_x = original_pixel_coords[:, 1] / original_width  # col / width
    relative_y = original_pixel_coords[:, 0] / original_height  # row / height
    
    relative_coords = np.column_stack([relative_y, relative_x])  # Keep (row, col) format
    
    # Calculate relative patch size (use minimum dimension for square patches)
    relative_patch_size = actual_patch_size_pixels / min(original_width, original_height)
    
    return relative_coords, relative_patch_size


class BNHead(nn.Module):
    """Batch Normalization Head for segmentation, based on DINOv2's BNHead."""
    
    def __init__(self, in_channels: int, num_classes: int, input_transform: str = "resize_concat"):
        super().__init__()
        self.input_transform = input_transform
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
    def _transform_inputs(self, inputs):
        """Transform inputs according to input_transform setting."""
        if self.input_transform == "resize_concat":
            if isinstance(inputs, (list, tuple)) and len(inputs) > 1:
                # Extract feature tensors from (tensor, class_token) tuples
                feature_tensors = [inp[0] for inp in inputs]
                
                # Get target size from first tensor (should be the largest)
                target_size = feature_tensors[0].shape[-2:]
                
                # Resize all tensors to target size and concatenate
                resized_inputs = []
                for tensor in feature_tensors:
                    if tensor.shape[-2:] != target_size:
                        resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
                        resized_inputs.append(resized)
                    else:
                        resized_inputs.append(tensor)
                
                # Concatenate along channel dimension
                return torch.cat(resized_inputs, dim=1)
            else:
                # Single input - extract tensor from (tensor, class_token) tuple
                inp = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                if isinstance(inp, (list, tuple)):
                    return inp[0]  # Extract tensor from (tensor, class_token)
                else:
                    return inp
        elif self.input_transform == "single_layer":
            # Linear head - use only the last feature level (highest resolution)
            if isinstance(inputs, (list, tuple)):
                last_feature = inputs[-1]  # Take last feature level
                if isinstance(last_feature, (list, tuple)):
                    return last_feature[0]  # Extract tensor from (tensor, class_token)
                else:
                    return last_feature
            else:
                return inputs
        else:
            # For other transform types, just take first tensor
            inp = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            if isinstance(inp, (list, tuple)):
                return inp[0]  # Extract tensor from (tensor, class_token)
            else:
                return inp
        
    def forward(self, x):
        """Forward pass through BN head."""
        # Transform inputs (handle multiple feature levels)
        x = self._transform_inputs(x)
        
        # Apply batch norm
        x = self.bn(x)
        
        # Apply final conv for classification
        x = self.conv_seg(x)
        
        return x


class DINOv2PatchExtractor:
    """DINOv2 patch feature extractor with smart resize."""
    
    def __init__(self, model_name: str = "dinov2_vitl14_ld", device: str = "cuda", image_size: int = 518, 
                 enable_depth: bool = False, depth_dataset: str = "nyu", 
                 enable_segmentation: bool = False, segmentation_dataset: str = "ade20k", 
                 segmentation_head_type: str = "linear"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.patch_size = 14  # DINOv2 standard patch size
        self.enable_depth = enable_depth
        self.depth_dataset = depth_dataset
        self.enable_segmentation = enable_segmentation
        self.segmentation_dataset = segmentation_dataset
        self.segmentation_head_type = segmentation_head_type
        
        # Load model with built-in depth head
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load segmentation head if enabled
        self.segmentation_head = None
        if self.enable_segmentation:
            self.segmentation_head = self._load_segmentation_head()
            self.segmentation_head.to(self.device)
            self.segmentation_head.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Calculate patches per dimension
        self.patches_per_dim = self.image_size // self.patch_size
        
        # Calculate relative patch size in original image space
        self.relative_patch_size = 1.0 / self.patches_per_dim
    
    def _load_segmentation_head(self) -> BNHead:
        """Load segmentation head weights and create BNHead."""
        # Map model names to backbone names for URL construction
        backbone_map = {
            "dinov2_vits14": "vits14", "dinov2_vits14_ld": "vits14",
            "dinov2_vitb14": "vitb14", "dinov2_vitb14_ld": "vitb14", 
            "dinov2_vitl14": "vitl14", "dinov2_vitl14_ld": "vitl14",
            "dinov2_vitg14": "vitg14", "dinov2_vitg14_ld": "vitg14"
        }
        
        backbone_name = backbone_map.get(self.model_name, "vitb14")
        head_type = "linear" if self.segmentation_head_type == "linear" else "ms"
        
        # Construct download URL
        url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{backbone_name}/dinov2_{backbone_name}_{self.segmentation_dataset}_{head_type}_head.pth"
        
        # Setup cache directory
        cache_dir = Path.home() / ".cache" / "dinov2_segmentation"
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"dinov2_{backbone_name}_{self.segmentation_dataset}_{head_type}_head.pth"
        
        # Download if not exists
        if not local_path.exists():
            print(f"Downloading segmentation head: {url}")
            urllib.request.urlretrieve(url, local_path)
            
        # Load checkpoint
        checkpoint = torch.load(local_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Determine input channels and num classes from weights
        # Look for decode_head.conv_seg.weight to get dimensions
        conv_seg_weight = None
        bn_weight = None
        
        for key, tensor in state_dict.items():
            if 'decode_head.conv_seg.weight' in key:
                conv_seg_weight = tensor
            elif 'decode_head.bn.weight' in key:
                bn_weight = tensor
                
        if conv_seg_weight is None:
            raise ValueError("Could not find conv_seg weights in checkpoint")
            
        num_classes = conv_seg_weight.shape[0]
        in_channels = bn_weight.shape[0]  # BN weight shows the actual concatenated input channels
        
        print(f"Creating segmentation head: {in_channels} -> {num_classes} classes")
        
        # Determine input transform type based on head type
        input_transform = "resize_concat" if head_type == "ms" else "single_layer"
        
        # Create BNHead
        segmentation_head = BNHead(in_channels=in_channels, num_classes=num_classes, input_transform=input_transform)
        
        # Load weights - need to map from state_dict keys to our model
        head_state_dict = {}
        for key, value in state_dict.items():
            if 'decode_head.bn' in key:
                new_key = key.replace('decode_head.', '')
                head_state_dict[new_key] = value
            elif 'decode_head.conv_seg' in key:
                new_key = key.replace('decode_head.', '')
                head_state_dict[new_key] = value
                
        segmentation_head.load_state_dict(head_state_dict)
        return segmentation_head
    
    def extract_patch_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract patch features with optional depth and segmentation from image with smart resize.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (patch_features, patch_coordinates, relative_patch_size, depth_output, segmentation_output)
            - patch_features: (n_valid_patches, feature_dim)
            - patch_coordinates: (n_valid_patches, 2) with relative (row, col) positions in [0,1]
            - relative_patch_size: Float representing patch size relative to image dimensions
            - depth_output: (height, width) depth map or None if depth disabled
            - segmentation_output: (height, width) segmentation map or None if segmentation disabled
        """
        # Smart resize with padding
        resized_image, padding_info = smart_resize_with_padding(image, self.image_size)
        
        # Preprocess for model
        image_tensor = self.transform(resized_image).unsqueeze(0).to(self.device)
        
        # Extract features based on model type, optionally extract depth
        with torch.no_grad():
            if '_ld' in self.model_name or '_dd' in self.model_name:
                # DepthEncoderDecoder model - extract features from backbone
                # res = self.model.whole_inference(image_tensor, img_meta=None, rescale=False)
                if "_dd" in self.model_name:
                    raw_backbone_features = self.model.extract_feat(image_tensor)
                else:
                    raw_backbone_features = self.model.backbone(image_tensor)
                
                x = raw_backbone_features[-1][0]  # Shape: (1, n_patches, feature_dim)
                # print("X", x.shape)
                patch_features = self.model.backbone.norm(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))
                # print("PATCHES", patch_features.shape)
                if self.enable_depth:
                    out = self.model._decode_head_forward_test(raw_backbone_features, None)
                    # crop the pred depth to the certain range.
                    depth_map = torch.clamp(out, 
                                                     min=self.model.decode_head.min_depth, 
                                                     max=self.model.decode_head.max_depth)
                    # Crop depth map back to original image size
                    depth_output = crop_depth_to_original_size(depth_map, padding_info, image.size, self.image_size)
                else:
                    depth_output = None
                
                # Extract segmentation if requested
                if self.enable_segmentation and self.segmentation_head is not None:
                    seg_output = self.segmentation_head(raw_backbone_features)
                    # Take argmax to get class predictions
                    seg_output = torch.argmax(seg_output, dim=1, keepdim=True)
                    # Crop segmentation map back to original image size (reuse depth function with nearest interpolation)
                    segmentation_output = crop_depth_to_original_size(seg_output.float(), padding_info, image.size, self.image_size, mode='nearest')
                else:
                    segmentation_output = None
                    
            else:
                # Base DINOv2 model - use standard approach
                model_output = self.model(image_tensor)
                if isinstance(model_output, dict):
                    patch_features = model_output['x_norm_patchtokens']  # Shape: (1, n_patches, feature_dim)
                else:
                    patch_features = model_output
                    
                # Base models don't support depth
                depth_output = None
                segmentation_output = None
        
        # Remove batch dimension from features
        patch_features = patch_features.squeeze(0)  # Shape: (n_patches, feature_dim)
        
        # Generate patch coordinates (row, col)
        patch_coords = np.array([
            [i, j] for i in range(self.patches_per_dim) 
            for j in range(self.patches_per_dim)
        ])
        
        # Filter out patches that contain padding
        valid_mask = filter_padded_patches(patch_coords, padding_info, self.patch_size, self.image_size)
        
        # Apply mask to get valid patches
        valid_patch_features = patch_features[valid_mask].cpu().numpy()
        valid_patch_coords = patch_coords[valid_mask]
                
        # Convert to relative coordinates in original image space
        relative_coords, relative_patch_size = convert_to_relative_coordinates(
            valid_patch_coords, padding_info, image.size, self.patch_size
        )
        
        return valid_patch_features, relative_coords, relative_patch_size, depth_output


def aggregate_depth_to_patches(depth_map: np.ndarray, patch_coords: np.ndarray, relative_patch_size: float, aggregation: str = 'median') -> np.ndarray:
    """
    Aggregate depth map values to per-patch statistics using patch coordinates.
    
    Args:
        depth_map: Depth map as numpy array (height, width)
        patch_coords: Patch coordinates in original image space (n_patches, 2) with (row, col) relative positions [0,1]
        relative_patch_size: Relative patch size (fraction of minimum image dimension)
        aggregation: Aggregation method - 'median', 'mean', 'min', 'max'
        
    Returns:
        Array of aggregated depth values per patch (n_patches,)
    """
    height, width = depth_map.shape
    
    # Calculate actual patch size in pixels based on image dimensions
    min_dimension = min(height, width)
    patch_size_pixels = relative_patch_size * min_dimension
    
    patch_depths = []
    
    for patch_coord in patch_coords:
        # Convert relative coordinates to pixel coordinates (use proper rounding)
        center_row = round(patch_coord[0] * height)
        center_col = round(patch_coord[1] * width)
        
        # Calculate patch boundaries with proper rounding
        half_patch = patch_size_pixels / 2
        start_row = max(0, round(center_row - half_patch))
        end_row = min(height, round(center_row + half_patch))
        start_col = max(0, round(center_col - half_patch))
        end_col = min(width, round(center_col + half_patch))
        
        # Extract patch region
        patch_region = depth_map[start_row:end_row, start_col:end_col]
        
        # Apply aggregation
        if patch_region.size > 0:
            if aggregation == 'median':
                patch_depth = np.median(patch_region)
            elif aggregation == 'mean':
                patch_depth = np.mean(patch_region)
            elif aggregation == 'min':
                patch_depth = np.min(patch_region)
            elif aggregation == 'max':
                patch_depth = np.max(patch_region)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        else:
            patch_depth = 0.0  # Fallback for empty regions
            
        patch_depths.append(patch_depth)
    
    return np.array(patch_depths)


def crop_depth_to_original_size(depth_map: torch.Tensor, padding_info: dict, original_size: tuple[int, int], input_size: int = 518, mode: str = 'bilinear') -> np.ndarray:
    """
    Crop tensor output back to original image size, removing padding and resizing.
    
    Args:
        depth_map: Output tensor from model (1, 1, H, W) or (1, H, W)
        padding_info: Padding information from smart_resize_with_padding
        original_size: (width, height) of original image
        input_size: Size of the padded input image (default 518)
        mode: Interpolation mode ('bilinear' for depth, 'nearest' for segmentation)
        
    Returns:
        Output as numpy array with shape (original_height, original_width)
    """
    # Ensure depth map is 4D for interpolation: (1, 1, H, W)
    if depth_map.dim() == 3:
        depth_map = depth_map.unsqueeze(1)  # Add channel dimension
    if depth_map.dim() == 2:
        depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # First, resize tensor to match the padded input size
    upsampled_depth = F.interpolate(
        depth_map, 
        size=(input_size, input_size), 
        mode=mode, 
        align_corners=False if mode == 'bilinear' else None
    )
    
    # Remove batch and channel dimensions for cropping: (H, W)
    upsampled_depth = upsampled_depth.squeeze(0).squeeze(0)
    
    # Remove padding
    top = padding_info['top']
    bottom = upsampled_depth.shape[0] - padding_info['bottom']
    left = padding_info['left'] 
    right = upsampled_depth.shape[1] - padding_info['right']
    
    cropped_depth = upsampled_depth[top:bottom, left:right]
    
    # Resize back to original image size
    original_width, original_height = original_size
    
    # Add batch and channel dimensions for final interpolation
    cropped_depth = cropped_depth.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Resize to original dimensions
    resized_depth = F.interpolate(
        cropped_depth, 
        size=(original_height, original_width), 
        mode=mode, 
        align_corners=False if mode == 'bilinear' else None
    )
    
    # Remove batch and channel dimensions, convert to numpy
    return resized_depth.squeeze(0).squeeze(0).cpu().numpy()


