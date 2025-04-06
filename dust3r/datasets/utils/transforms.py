# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torchvision.transforms as tvf

import torch
import numpy as np
import cv2
import random
import math
from typing import Dict, List, Tuple, Union, Optional, Callable
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image

# define the standard image transforms
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])
PhotometricTransforms = tvf.Compose(
    [
        tvf.ColorJitter(0.5, 0.5, 0.5, 0.1),
        tvf.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ImgNorm
    ]
)

class CorrespondenceRandomRotation(torch.nn.Module):
    """Apply random rotation to images and corresponding points."""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]], p: float = 0.5):
        """
        Args:
            degrees: Range of rotation degrees. If float, rotates between (-degrees, +degrees).
                    If tuple, rotates between degrees[0] and degrees[1].
            p: Probability of applying the transform.
        """
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.p = p
        
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
            
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        transformed_images = []
        transformed_keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Handle PIL Image or Tensor
            if isinstance(img, Image.Image):
                width, height = img.size
                # Convert to tensor for consistency in returned type
                img_tensor = F.to_tensor(img)
                # Rotate image
                rotated_img = F.rotate(img, angle=angle)
                transformed_images.append(rotated_img)
            else:  # Tensor
                if img.ndim == 3:
                    height, width = img.shape[1], img.shape[2]
                else:
                    height, width = img.shape
                rotated_img = F.rotate(img, angle=angle)
                transformed_images.append(rotated_img)
                
            # Convert to numpy for point transformation
            angle_rad = math.radians(-angle)  # Negative because PIL rotates counter-clockwise
            cos_val = math.cos(angle_rad)
            sin_val = math.sin(angle_rad)
            
            # Calculate center
            cx, cy = width / 2, height / 2
            
            # Transform points
            transformed_points = []
            for x, y in points:
                # Translate to origin
                x -= cx
                y -= cy
                
                # Rotate
                new_x = x * cos_val - y * sin_val
                new_y = x * sin_val + y * cos_val
                
                # Translate back
                new_x += cx
                new_y += cy
                
                transformed_points.append([new_x, new_y])
                
            transformed_keypoints.append(torch.tensor(transformed_points))
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceRandomHorizontalFlip(torch.nn.Module):
    """Apply random horizontal flip to images and corresponding points."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the transform.
        """
        super().__init__()
        self.p = p
        
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
            
        transformed_images = []
        transformed_keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Handle PIL Image or Tensor
            if isinstance(img, Image.Image):
                width, height = img.size
                flipped_img = F.hflip(img)
            else:  # Tensor
                if img.ndim == 3:
                    height, width = img.shape[1], img.shape[2]
                else:
                    height, width = img.shape
                flipped_img = F.hflip(img)
            
            transformed_images.append(flipped_img)
            
            # Transform points
            transformed_points = []
            for x, y in points:
                # Flip x-coordinate
                new_x = width - x
                transformed_points.append([new_x, y])
                
            transformed_keypoints.append(torch.tensor(transformed_points))
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceRandomVerticalFlip(torch.nn.Module):
    """Apply random vertical flip to images and corresponding points."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying the transform.
        """
        super().__init__()
        self.p = p
        
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
            
        transformed_images = []
        transformed_keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Handle PIL Image or Tensor
            if isinstance(img, Image.Image):
                width, height = img.size
                flipped_img = F.vflip(img)
            else:  # Tensor
                if img.ndim == 3:
                    height, width = img.shape[1], img.shape[2]
                else:
                    height, width = img.shape
                flipped_img = F.vflip(img)
            
            transformed_images.append(flipped_img)
            
            # Transform points
            transformed_points = []
            for x, y in points:
                # Flip y-coordinate
                new_y = height - y
                transformed_points.append([x, new_y])
                
            transformed_keypoints.append(torch.tensor(transformed_points))
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceRandomScale(torch.nn.Module):
    """Apply random scaling to images and corresponding points."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        """
        Args:
            scale_range: Range of scaling factors.
            p: Probability of applying the transform.
        """
        super().__init__()
        self.scale_range = scale_range
        self.p = p
        
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
            
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        transformed_images = []
        transformed_keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Handle PIL Image or Tensor
            if isinstance(img, Image.Image):
                width, height = img.size
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_img = F.resize(img, [new_height, new_width])
                
                # Resize back to original size to keep dimensions consistent
                scaled_img = F.resize(resized_img, [height, width])
            else:  # Tensor
                if img.ndim == 3:
                    height, width = img.shape[1], img.shape[2]
                else:
                    height, width = img.shape
                    
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_img = F.resize(img, [new_height, new_width])
                
                # Resize back to original size
                scaled_img = F.resize(resized_img, [height, width])
            
            transformed_images.append(scaled_img)
            
            # Transform points - scale from center
            transformed_points = []
            cx, cy = width / 2, height / 2
            
            for x, y in points:
                # Scale relative to center
                new_x = (x - cx) * scale + cx
                new_y = (y - cy) * scale + cy
                
                # Adjust for resize back to original dimensions
                new_x = new_x * width / new_width
                new_y = new_y * height / new_height
                
                transformed_points.append([new_x, new_y])
                
            transformed_keypoints.append(torch.tensor(transformed_points))
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceRandomTranslation(torch.nn.Module):
    """Apply random translation to images and corresponding points."""
    
    def __init__(self, max_fraction: float = 0.1, p: float = 0.5):
        """
        Args:
            max_fraction: Maximum translation as fraction of image dimensions.
            p: Probability of applying the transform.
        """
        super().__init__()
        self.max_fraction = max_fraction
        self.p = p
        
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
        
        # Get the size of the first image to determine translation range
        img = sample['images'][0]
        if isinstance(img, Image.Image):
            width, height = img.size
        else:  # Tensor
            if img.ndim == 3:
                height, width = img.shape[1], img.shape[2]
            else:
                height, width = img.shape
        
        # Choose translation within the max_fraction range
        max_dx = int(width * self.max_fraction)
        max_dy = int(height * self.max_fraction)
        dx = random.randint(-max_dx, max_dx)
        dy = random.randint(-max_dy, max_dy)
        
        # We need to perform translate using affine transform since F.translate
        # has different behavior for PIL and Tensor
        transformed_images = []
        transformed_keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Create affine matrix for translation
            affine_matrix = torch.tensor([
                [1, 0, dx],
                [0, 1, dy]
            ], dtype=torch.float)
            
            # Apply transform to image
            if isinstance(img, Image.Image):
                # Convert PIL to tensor first
                img_tensor = F.to_tensor(img)
                translated_img = F.affine(
                    img_tensor, 
                    angle=0, 
                    translate=[dx, dy], 
                    scale=1.0, 
                    shear=[0, 0]
                )
                transformed_images.append(translated_img)
            else:
                translated_img = F.affine(
                    img, 
                    angle=0, 
                    translate=[dx, dy], 
                    scale=1.0, 
                    shear=[0, 0]
                )
                transformed_images.append(translated_img)
            
            # Transform points
            transformed_points = []
            for x, y in points:
                new_x = x + dx
                new_y = y + dy
                transformed_points.append([new_x, new_y])
                
            transformed_keypoints.append(torch.tensor(transformed_points))
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceRandomCrop(torch.nn.Module):
    """Apply random crop to images and corresponding points."""
    
    def __init__(self, 
                 size: Union[int, Tuple[int, int]], 
                 scale: Tuple[float, float] = (0.7, 1.0),
                 p: float = 0.5,
                 filter_keypoints: bool = True):
        """
        Args:
            size: Desired output size (h, w) of the crop. If int, square crop is applied.
            scale: Range of size of the origin crop
            p: Probability of applying the transform
            filter_keypoints: Whether to filter out keypoints that fall outside the crop
        """
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.p = p
        self.filter_keypoints = filter_keypoints
        
    def get_params(self, img: Union[torch.Tensor, Image.Image], scale: Tuple[float, float]) -> Tuple[int, int, int, int]:
        """Get parameters for crop."""
        if isinstance(img, Image.Image):
            width, height = img.size
        else:  # Tensor
            if img.ndim == 3:
                height, width = img.shape[1], img.shape[2]
            else:
                height, width = img.shape
                
        # Choose a random scale factor
        scale_factor = random.uniform(scale[0], scale[1])
        
        # Compute crop dimensions
        crop_height = int(height * scale_factor)
        crop_width = int(width * scale_factor)
        
        # Choose random crop location
        if crop_width > width:
            crop_width = width
        if crop_height > height:
            crop_height = height
            
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        return top, left, crop_height, crop_width
    
    def forward(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        if random.random() > self.p:
            return sample
        
        # Get crop parameters using the first image
        top, left, height, width = self.get_params(sample['images'][0], self.scale)
        
        transformed_images = []
        transformed_keypoints = []
        all_valid_indices = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            # Crop image
            if isinstance(img, Image.Image):
                cropped_img = F.crop(img, top, left, height, width)
                # Resize to desired output size
                cropped_img = F.resize(cropped_img, self.size)
                transformed_images.append(cropped_img)
            else:  # Tensor
                cropped_img = F.crop(img, top, left, height, width)
                cropped_img = F.resize(cropped_img, list(self.size))
                transformed_images.append(cropped_img)
            
            # Transform points
            if not isinstance(points, torch.Tensor):
                points = torch.tensor(points)
                
            # Adjust points to the cropped region
            adjusted_points = points.clone()
            adjusted_points[:, 0] = points[:, 0] - left
            adjusted_points[:, 1] = points[:, 1] - top
            
            # Scale points to the new size
            if height > 0 and width > 0:  # Avoid division by zero
                scale_x = self.size[1] / width
                scale_y = self.size[0] / height
                
                adjusted_points[:, 0] = adjusted_points[:, 0] * scale_x
                adjusted_points[:, 1] = adjusted_points[:, 1] * scale_y
            
            if self.filter_keypoints:
                # Identify points within the crop
                valid_indices = (
                    (points[:, 0] >= left) & 
                    (points[:, 0] < left + width) & 
                    (points[:, 1] >= top) & 
                    (points[:, 1] < top + height)
                )
                
                # Store indices of valid keypoints
                all_valid_indices.append(valid_indices)
                
            transformed_keypoints.append(adjusted_points)
        
        # If filtering keypoints, ensure we only keep corresponding points that are valid in all images
        if self.filter_keypoints:
            # Find common valid indices across all images
            common_valid = all_valid_indices[0]
            for valid in all_valid_indices[1:]:
                common_valid = common_valid & valid
                
            # If we have any valid points, filter all keypoint sets
            if common_valid.any():
                filtered_keypoints = []
                for points in transformed_keypoints:
                    filtered_keypoints.append(points[common_valid])
                transformed_keypoints = filtered_keypoints
            else:
                # If no valid points remain, return the original sample
                return sample
        
        return {
            'images': transformed_images,
            'keypoints': transformed_keypoints
        }


class CorrespondenceCompose(object):
    """Compose multiple transforms that handle image-point correspondences."""
    
    def __init__(self, transforms: List[Callable]):
        """
        Args:
            transforms: List of transforms to compose.
        """
        self.transforms = transforms
        
    def __call__(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images or tensors
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Transformed sample
        """
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert PIL Images to tensors and ensure keypoints are tensors."""
    
    def __call__(self, sample: Dict) -> Dict:
        """
        Args:
            sample: Dictionary containing:
                - 'images': List of PIL Images
                - 'keypoints': List of point arrays [N, 2]
                
        Returns:
            Sample with tensor images and keypoints
        """
        images = []
        keypoints = []
        
        for img, points in zip(sample['images'], sample['keypoints']):
            if isinstance(img, Image.Image):
                # Convert PIL Image to tensor
                img_tensor = F.to_tensor(img)
                images.append(img_tensor)
            else:
                # Already a tensor
                images.append(img)
            
            # Ensure keypoints are tensors
            if not isinstance(points, torch.Tensor):
                points_tensor = torch.tensor(points, dtype=torch.float)
                keypoints.append(points_tensor)
            else:
                keypoints.append(points)
        
        return {
            'images': images,
            'keypoints': keypoints
        }

