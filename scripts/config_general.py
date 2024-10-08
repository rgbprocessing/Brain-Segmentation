# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:26:03 2024

@author: Dragon
"""

# Data configurations
DATA_SHAPE = (2, 192, 192, 192)
LABELS_SHAPE = (7, 192, 192, 192)
DATA_FILENAME = "data.pt"
LABELS_FILENAME = "mask1B2G3P4L5W6E7V.pt"

# Patch configuration
PATCH_DATA_SHAPE = (2, 32, 32, 32)
PATCH_LABELS_SHAPE = (7, 32, 32, 32)
NUM_CLASSES_SAMPLE = 6                      #number of classes to sample from [0,NUM_CLASSES_SAMPLE] inclusive

# Augmentation configurations
GAUSS_NOISE = 1/50
RANDOM_NOISE = 1/50

