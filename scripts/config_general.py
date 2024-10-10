# -*- coding: utf-8 -*-
# config_general.py
# Configuration File for Brain MRI Segmentation Project
#
# This file contains various configuration parameters used throughout the project,
# including data paths, model parameters, training settings, and data processing
# constants. Modify these values as needed for different experiments or setups.. 
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project

# Train configuration
TRNSET = [1,3,4]
VALSET = [2]
TSTSET = [5]
DFPOOL_FILE = r'C:\Users\Dragon\Documents\GitHub\Brain-Segmentation\class_sums.xlsx'

# Specify training parameters patches
NUM_EPOCHS = 400
IN_CHANNELS = 2 #modalities
OUT_CHANNELS = 7
LR = 0.001
BATCH_SIZE = 80 #TODO
NUM_SAMPLES = 80*16#TODO
PATCH_SIZE = 32

# Model parameters
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)
RES_UNITS = 2
NORM = "batch"
DROPOUT = .1

# Data configurations
DATA_SHAPE = (2, 192, 192, 192)             # Full brain data shape
LABELS_SHAPE = (7, 192, 192, 192)           # Full brain labels shape
DATA_FILENAME = "data.pt"                   # Data filename
LABELS_FILENAME = "mask1B2G3P4L5W6E7V.pt"   # Labels filename

# Patch configuration
PATCH_DATA_SHAPE = (2, 32, 32, 32)          # Final shape of data patches
PATCH_LABELS_SHAPE = (7, 32, 32, 32)        # Final shape of label patches
NUM_CLASSES_SAMPLE = 6                      # Number of classes to sample from [0,NUM_CLASSES_SAMPLE] inclusive

# Augmentation configurations
GAUSS_NOISE = 1/50                          # Standard deviation for Gaussian noise
RANDOM_NOISE = 1/50                         # Standard deviation for random noise
