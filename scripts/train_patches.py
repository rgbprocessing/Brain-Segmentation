# -*- coding: utf-8 -*-
# train_patches.py
# This script implements a patch-based training approach for 3D brain MRI 
# segmentation. 
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project
# Requires: numpy, os, torch, monai, pandas, tqdm
# 
# Usage:
# - Ensure all required libraries are installed
# - Set the appropriate data paths and hyperparameters in the config section
# - Run the script to start the patch-based training process
#
# Note: This script is part of a larger brain segmentation project. For full 
# context and additional components, please refer to the project's documentation.

from config_patches import *
from utils.data_utils import EqualProbabilityRandFlipd, getCumulativeAry
from utils.train_utils import create_incremented_run_folder, create_run_folder

import torch
import numpy as np
import monai
from monai.data import DataLoader
from monai.networks.nets import UNet, UNETR, SwinUNETR, DynUNet
from monai.transforms import AsDiscrete, Compose, RandAffined, EnsureTyped
from monai.metrics import ConfusionMatrixMetric, DiceMetric
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
import warnings
import gc
import os

from dataset_patches import BrainMRIDataset
from dataset_fullbrain import BrainMRIDataset as BrainMRIDatasetTEST

def main():
    # Create new logging directory
    run_dir, checkpoints_dir, logs_dir = create_run_folder()
    
    # Create new TensorBoard writer instance
    writer = SummaryWriter(log_dir=logs_dir)
    
    # Load the Excel file into a pandas DataFrame
    dfpool = pd.read_excel(DFPOOL_FILE)
    
    dftrn = dfpool[dfpool['Fold'].isin(TRNSET)]
    dfval = dfpool[dfpool['Fold'].isin(VALSET)]
    dftst = dfpool[dfpool['Fold'].isin(TSTSET)]
    
    df_new_trn = pd.concat([dftrn, pd.DataFrame(getCumulativeAry(dftrn),index=dftrn.index,columns=[10,11,12,13,14,15,16])], axis=1)
    df_new_tst = pd.concat([dftst, pd.DataFrame(getCumulativeAry(dftst),index=dftst.index,columns=[10,11,12,13,14,15,16])], axis=1)
    df_new_val = pd.concat([dfval, pd.DataFrame(getCumulativeAry(dfval),index=dfval.index,columns=[10,11,12,13,14,15,16])], axis=1)
    
    trn_list = dftrn['Filepath'].to_list()
    val_list = dfval['Filepath'].to_list()
    tst_list = dftst['Filepath'].to_list()
    
    # Set up CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
       
    # Initialize model
    model = UNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS, 
        strides=STRIDES, 
        num_res_units=RES_UNITS,
        norm=NORM,
        dropout = DROPOUT
    )
      
    # Define data transforms
    train_transforms = Compose([
        EqualProbabilityRandFlipd(keys=["image", "label"], spatial_axis=(1,2,3)),
        RandAffined(
            keys=["image", "label"],
            prob=TRANSFORM_PROB,
            rotate_range=(2*np.pi, 2*np.pi, 2*np.pi),  # Full rotation range
            scale_range=(0.1, 0.1, 0.1),
            shear_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            spatial_size=(101, 101, 101),  # Adjust to your image size
            padding_mode="zeros",
        ),
        EnsureTyped(keys=["image", "label"])
    ])
    
    with torch.cuda.device(device):
        model = model.to(device) 
        
        # Prepare datasets
        train_dataset = BrainMRIDataset(df_new_trn, PATCH_SIZE, NUM_SAMPLES, rotations=True, gauss=GAUSS_NOISE, noise=RANDOM_NOISE, transform=train_transforms)
        val_dataset = BrainMRIDatasetTEST(val_list)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8,pin_memory=True,prefetch_factor=10)
        val_loader = DataLoader(val_dataset, batch_size=3,num_workers=8,pin_memory=True,prefetch_factor=10)
        
        # Define loss function
        diceCE_loss = monai.losses.DiceCELoss(to_onehot_y=False, softmax=True, include_background=False, lambda_dice=0)
        focal_loss = monai.losses.FocalLoss(to_onehot_y=False)
        tversky_loss = monai.losses.TverskyLoss(to_onehot_y=False, softmax=True, include_background=False, alpha=.3, beta=.7)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Train the model
        train_model(model, train_loader, val_loader, diceCE_loss, tversky_loss, optimizer, num_epochs=NUM_EPOCHS, device=device, writer=writer)
          
if __name__ == '__main__':
    print('starting')
    main()