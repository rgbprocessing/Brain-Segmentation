# -*- coding: utf-8 -*-
# data_utils.py
# This module contains custom transforms and utility functions used in the 
# brain MRI segmentation project. It includes transformations for data 
# augmentation including a custom random flip transform with equal probability 
# for all flip combinations, a noise application function, and a function to 
# calculate cumulative arrays for data preparation.
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project
# Requires: torch, numpy, monai
# 
# Note: This script is part of a larger brain segmentation project. For full 
# context and additional components, please refer to the project's documentation.

import torch
from monai.transforms import Transform
import numpy as np

class EqualProbabilityRandFlipd(Transform):
    """
    A custom transform that applies random flips with equal probability for all 
    combinations of specified axes.
    """
    def __init__(self, keys, spatial_axis=(1, 2, 3)):
        """
        Initialize the EqualProbabilityRandFlipd transform.

        Args:
            keys (list): List of keys to apply the transform to.
            spatial_axis (tuple): Axes to consider for flipping. Default is (1, 2, 3) for 3D data.
        """
        self.keys = keys
        self.spatial_axis = spatial_axis
        self.num_combinations = 2 ** len(spatial_axis)

    def __call__(self, data):
        """
        Apply the random flip transform to the input data.

        Args:
            data (dict): Input data dictionary containing image and label tensors.

        Returns:
            dict: Transformed data dictionary.

        Raises:
            TypeError: If the data for any key is not a torch.Tensor.
        """
        flip_choice = torch.randint(self.num_combinations, (1,)).item()
        flip_axes = [axis for i, axis in enumerate(self.spatial_axis) if flip_choice & (1 << i)]
        
        for key in self.keys:
            if torch.is_tensor(data[key]):
                data[key] = torch.flip(data[key], dims=flip_axes)
            else:
                raise TypeError(f"Data with key {key} is not a torch.Tensor")
        
        return data
    
def apply_noise(data_dict, noise, gauss):
    """
    Apply random noise and Gaussian noise to the image data.

    Args:
        data_dict (dict): Input data dictionary containing 'image' and 'label' keys.
        noise (float): Magnitude of random noise to apply.
        gauss (float): Standard deviation of Gaussian noise to apply.

    Returns:
        dict: Data dictionary with noise applied to the image.
    """
    image = data_dict['image']
    label = data_dict['label']
    
    # Generate noise factors
    noiseA = torch.randn(1, dtype=image.dtype) * noise
    noiseB = torch.randn(1, dtype=image.dtype) * noise

    # Apply noise
    image = image * ( 1 + noiseA) + noiseB

    # Add Gaussian noise
    gaussian_noise = torch.randn(image.shape, dtype=image.dtype) * gauss
    image = image + gaussian_noise
    
    new_dict = {
        'image': image,
        'label': label,
    }
    
    return new_dict

# create empty dataframe for cumulative values
def getCumulativeAry(df):
    """
    Calculate cumulative arrays for each class in the input DataFrame.

    This function is used for data preprocessing to create cumulative counts
    of voxels for each class across all samples.

    Args:
        df (pandas.DataFrame): Input DataFrame containing class-wise voxel counts.

    Returns:
        numpy.ndarray: Array of cumulative voxel counts for each class.
    """
    cumulativeary = np.zeros([len(df),7])

    for i in range(2,9):
        c=0
        # calculate cumulative values
        cumulative = 0
        for index, row in df.iterrows():
            cumulative += row[i]
            cumulativeary[c,i-2]=int(cumulative)
            c+=1

    cumulativeary=cumulativeary.astype(int)

    return cumulativeary