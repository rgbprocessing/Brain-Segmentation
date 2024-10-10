# -*- coding: utf-8 -*-
# dataset_patches.py
# This module defines a custom PyTorch Dataset for loading and preprocessing
# brain MRI data patches for segmentation tasks. It includes the ability to do 
# full rotation augmentation in all dimensions by first selecting a larger
# region, then performing transforms, and finally cropping down to the desired
# patch size. The pre-configured patch sizes are 101 for the preliminary crop
# and 32  for the final patch size. 101 is calculated to allow for random 
# rotation around any axis and scaling up to 10% increase in size.
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project
# Requires: os, torch, numpy, random
# 
# Usage:
# - from dataset_fullbrain import BrainMRIDataset
# - train_dataset = BrainMRIDataset(df_new_trn, patch_size, num_samples, label_size, rotations=True, gauss=1/50, noise=1/50, transform=train_transforms)
#
# Note: This script is part of a larger brain segmentation project. For full 
# context and additional components, please refer to the project's documentation.

from config_general import *
from utils.data_utils import apply_noise
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class BrainMRIDataset(Dataset):
    """
    A custom Dataset class for loading and preprocessing brain MRI data.
    This class handles data loading, patch extraction, and data augmentation.
    """
    
    def __init__(self, df_new, patch_size, num_samples, rotations=True, gauss=1/50, noise=1/50, transform=None): 
        """
        Initialize the BrainMRIDataset.

        Args:
            df_new (DataFrame): DataFrame containing file paths and metadata.
            patch_size (int): Size of the patches to extract.
            num_samples (int): Number of samples to generate.
            label_size (int): Size of the label patches.
            rotations (bool): Whether to apply rotations. Default is True.
            gauss (float): Gaussian noise factor. Default is 1/50
            noise (float): Random noise factor. Default is 1/50
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patch_size = patch_size
        self.rotations = rotations
        self.df_new=df_new
        self.gauss = gauss
        self.noise = noise
        self.num_samples = num_samples
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples generated"""
        return self.num_samples
    
    def __getitem__(self, index):
        """
           Get a single item from the dataset.
    
           This method handles the core logic of data loading, patch extraction,
           and data augmentation for each sample.
    
           Args:
               index (int): Index of the item to retrieve.
    
           Returns:
               dict: A dictionary containing 'image' and 'label' data.
        """
        # Select a random class (0 to 6 inclusive)
        i = random.randint(0,NUM_CLASSES_SAMPLE)
        
        # Get total number of voxels for the selected class (handles class_sums format)
        total = int(self.df_new[-1:][NUM_CLASSES_SAMPLE+3+i+1])
        
        # Randomly select a sample within the class
        sample = random.randint(1,total)
        classrow = (self.df_new[9+i+1]>=sample).idxmax()
        
        # Find the associated brain MRI file
        sampledf = self.df_new['Filepath'][classrow]
        data= torch.load(os.path.join(sampledf,DATA_FILENAME))
        labels = torch.load(os.path.join(sampledf,LABELS_FILENAME))
        # sizes: [7, 192, 192, 192], [2, 192, 192, 192]
        
        # Find voxels of the selected class
        condition = (labels[i,:,:,:]==1)
        indices = torch.nonzero(condition)
        
        # Choose a random voxel from the selected class and brain file
        target_index = random.choice(indices)
        
        # Crop patches centered around the selected voxel
        # crop [101, 101, 101] - calculated to allow for a [32, 32, 32] crop with random rotation and scale *10%
        # centered around the selected voxel
        data = crop_patch(data, target_index)
        labels = crop_patch(labels, target_index)

        # Ensure data is in the correct format [2, 192, 192, 192]
        if data.shape != DATA_SHAPE:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Ensure labels are in the correct format [7, 192, 192, 192]
        if labels.shape != LABELS_SHAPE:
            raise ValueError(f"Unexpected labels shape: {labels.shape}")
            
        # Prepare the input dictionary
        dict_input = {
            'image': data,
            'label': labels
        }
        
        # Apply any specified transforms
        if self.transform:
            dict_input = self.transform(dict_input)
        
        # Apply second crop for final patch size
        dict_input = second_crop(dict_input)
        
        # Apply noise augmentation
        dict_input = apply_noise(dict_input, self.noise, self.gauss)
        
        # Ensure data is in the correct format [2, 192, 192, 192]
        if dict_input['image'].shape != PATCH_DATA_SHAPE:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Ensure labels are in the correct format [7, 192, 192, 192]
        if dict_input['label'].shape != PATCH_LABELS_SHAPE:
            raise ValueError(f"Unexpected labels shape: {labels.shape}")
        
        return dict_input

def crop_patch(tensor, center, patch_size=101):
    """
    Crop a 3D patch from a 4D tensor (C, D, H, W) centered around a given point.
    
    Args:
        tensor (torch.Tensor): Input 4D tensor (C, D, H, W)
        center (tuple): Center coordinates (x, y, z) for the patch
        patch_size (int): Size of the cubic patch to extract (default: 101)
    
    Returns:
        torch.Tensor: Cropped patch of shape (C, patch_size, patch_size, patch_size)
    """
    C, D, H, W = tensor.shape
    x, y, z = center
    
    # Add batch dimension for padding operation
    tensor = tensor.unsqueeze(0)
    
    # Calculate the boundaries of the patch
    x_start = x - patch_size // 2
    x_end = x_start + patch_size
    y_start = y - patch_size // 2
    y_end = y_start + patch_size
    z_start = z - patch_size // 2
    z_end = z_start + patch_size
    
    # Calculate required padding in each dimension
    pad_x = max(0, -x_start) + max(0, x_end - D)
    pad_y = max(0, -y_start) + max(0, y_end - H)
    pad_z = max(0, -z_start) + max(0, z_end - W)
    
    # Apply circular padding if necessary
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        tensor = F.pad(tensor, (max(0, -z_start), max(0, z_end - W),
                                max(0, -y_start), max(0, y_end - H),
                                max(0, -x_start), max(0, x_end - D)),
                       mode='circular') #'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    
    # Remove batch dimension
    tensor = tensor.squeeze(0)
    
    # Adjust start indices to ensure they are non-negative
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    z_start = max(0, z_start)
    
    # Crop the patch
    patch = tensor[:, x_start:x_start+patch_size, 
                   y_start:y_start+patch_size, 
                   z_start:z_start+patch_size]
    
    return patch

def second_crop(data_dict, patch_size=32):
    """
    Perform a second crop on both image and label data to extract a smaller patch.
    
    Args:
        data_dict (dict): Dictionary containing 'image' and 'label' tensors
        patch_size (int): Size of the cubic patch to extract (default: 32)
    
    Returns:
        dict: New dictionary with cropped 'image' and 'label' tensors
    """
    image = data_dict['image']
    label = data_dict['label']
    
    C, D, H, W = image.shape
    
    # Generate random offsets for cropping
    x = random.randint(0,patch_size-1)
    y = random.randint(0,patch_size-1)
    z = random.randint(0,patch_size-1)
    
    # Crop the image and label tensors using offset
    image = image[:, D//2 - x:D//2 - x+patch_size, H//2 - y:H//2 - y+patch_size, W//2 - z:W//2 - z+patch_size]
    label = label[:, D//2 - x:D//2 - x+patch_size, H//2 - y:H//2 - y+patch_size, W//2 - z:W//2 - z+patch_size]
    
    # Create a new dictionary with the cropped tensors
    new_dict = {
        'image': image,
        'label': label,
    }
    
    return new_dict