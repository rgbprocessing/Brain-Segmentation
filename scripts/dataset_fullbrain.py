# -*- coding: utf-8 -*-
# dataset_fullbrain.py
# This module defines a custom PyTorch Dataset for loading and preprocessing
# full brain MRI data for segmentation tasks. It includes data augmentation
# techniques such as Gaussian noise and random noise addition.
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project
# Requires: os, torch
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
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class BrainMRIDataset(Dataset):
    """
    A custom Dataset class for loading brain MRI data and labels.

    This class loads MRI data and corresponding segmentation labels,
    applies noise augmentation, and prepares the data for model input.

    Attributes:
        data_list (list): List of paths to data samples.
        gauss (float): Standard deviation for Gaussian noise.
        noise (float): Standard deviation for random noise.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
      
    def __init__(self, data_list, gauss=1/50, noise=1/50, transform=None): 
        """
        Initialize the dataset.

        Args:
            data_list (list): List of paths to data samples.
            gauss (float): Standard deviation for Gaussian noise. Default is 1/50.
            noise (float): Standard deviation for random noise. Default is 1/50.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_list = data_list
        self.gauss = gauss
        self.noise = noise
        self.transform = transform
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset.

        This method loads the MRI data and labels, applies noise augmentation,
        and returns the processed data and labels.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'image' and 'label' keys with the processed data.

        Raises:
            ValueError: If the loaded data or labels have unexpected shapes.
        """
        sampledf = self.data_list[index]
        
        # Load data and labels
        data= torch.load(os.path.join(sampledf,DATA_FILENAME))
        labels = torch.load(os.path.join(sampledf,LABELS_FILENAME))
        
        # Ensure data is in the correct format [2, 192, 192, 192]
        if data.shape != DATA_SHAPE:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Ensure labels are in the correct format [7, 192, 192, 192]
        if labels.shape != LABELS_SHAPE:
            raise ValueError(f"Unexpected labels shape: {labels.shape}")
        
        # Return a dictionary with 'image' and 'label' keys
        dict_input = {
            'image': data,
            'label': labels
        }
        
        # Apply additional transforms if specified
        if self.transform:
            dict_input = self.transform(dict_input)
            
        # Apply noise augmentation
        dict_input = apply_noise(dict_input, self.noise, self.gauss)
            
        return dict_input
    
