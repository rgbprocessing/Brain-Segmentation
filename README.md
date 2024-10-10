# Brain MRI Segmentation Project

## Overview
This project focuses on training and evaluating deep learning models for brain MRI segmentation. It supports both full brain and patch-based segmentation approaches, as well as inference and model ensembling.

**Note: This project is currently under active development and is being regularly updated.**

## Features
- Full brain segmentation model training
- Patch-based segmentation model training
- Inference pipeline for trained models
- Model ensembling capabilities
- Support for various data augmentation techniques
- TensorBoard integration for training visualization

## Project Structure
- `scripts/`: Main directory for all scripts
  - `dataset_fullbrain.py`: Dataset class for full brain segmentation
  - `dataset_patches.py`: Dataset class for patch-based segmentation
  - `train_patches.py`: Script for training patch-based models
  - `config_general.py`: General configuration file
  - `utils/`: Utility functions
    - `data_utils.py`: Data loading and preprocessing utilities
    - `train_utils.py`: Training loop and related functions

## Current Status
This project is actively being developed. Upcoming improvements include:
- Refactoring and optimizing the fullbrain training and inference codes
- Enhancing documentation and code comments
- Adding visualization notebook