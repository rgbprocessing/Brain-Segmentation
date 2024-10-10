# -*- coding: utf-8 -*-
# train_utils.py
# This module contains utility functions for managing training runs and the 
# main training loop for a brain MRI segmentation model. It includes 
# functionality for creating incremented run folders, setting up checkpoints 
# and logs, and the core training and validation process.
#
# Author: Dani
# Created: 2022-08-22
# Last Modified: 2024-10-08
#
# Part of the Brain Segmentation Project
# Requires: os, torch, monai
# 
# Note: This script is part of a larger brain segmentation project. For full 
# context and additional components, please refer to the project's documentation.

from ..config_general import *
import os
import torch
from tqdm import tqdm
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete

def create_incremented_run_folder(base_path="./runs"):
    """
    Create a new run folder with an incremented number.
    
    Args:
        base_path (str): Base directory for run folders.
    
    Returns:
        str: Path to the newly created run folder.
    """
    # Ensure the base directory exists
    os.makedirs(base_path, exist_ok=True)

    # Get all existing run folders
    existing_runs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("run")]

    if not existing_runs:
        # If no runs exist, start with run000
        new_run_name = "run000"
    else:
        # Find the highest numbered run
        highest_run = max(existing_runs)
        highest_num = int(highest_run[3:])  # Extract the number part
        new_run_name = f"run{highest_num + 1:03d}"  # Increment and zero-pad to 3 digits

    # Create the new run folder
    new_run_path = os.path.join(base_path, new_run_name)
    os.makedirs(new_run_path)

    print(f"Created new run folder: {new_run_path}")
    return new_run_path

def create_run_folder():
    """
    Create a new run folder with subdirectories for checkpoints and logs.
    
    Returns:
        tuple: Paths to run directory, checkpoints directory, and logs directory.
    """
    run_dir = create_incremented_run_folder(base_path="./runs_patches")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs", "tensorboard_logs")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return run_dir, checkpoints_dir, logs_dir

def train_model(model, train_loader, val_loader, criterion1, criterion2, optimizer, num_epochs, device, writer, checkpoints_dir):
    """
    Main training loop for the brain MRI segmentation model.
    
    Args:
        model (torch.nn.Module): The neural network model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion1, criterion2 (callable): Loss functions
        optimizer (torch.optim.Optimizer): Optimizer for model parameters
        num_epochs (int): Number of training epochs
        device (torch.device): Device to run the model on
        writer (SummaryWriter): TensorBoard SummaryWriter for logging
        checkpoints_dir (str): Directory to save model checkpoints
    """
    # Initialize metrics and variables
    train_dice_metric, val_dice_metric = initialize_metrics()
    best_loss, alpha, accumulation_steps = 100.00, LOSS_ALPHA, ACCUMULATION
    best_LI, best_LI_sens, best_LI_trn = 0, 0, 0
    post_pred = AsDiscrete(argmax=True, to_onehot=OUT_CHANNELS)

    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_dice_scores = train_epoch(model, train_loader, optimizer, criterion1, criterion2, 
                                                    alpha, train_dice_metric, post_pred, accumulation_steps, device)
        
        # Log training results
        log_results(writer, epoch, train_loss, train_dice_scores, "Train")
        
        # Validation phase (if LI dice score is above threshold)
        if train_dice_scores[0][3] > VAL_THRESHOLD:
            best_LI_trn = train_dice_scores[0][3].item()
            val_loss, val_dice_scores, sensitivity = validate_model(model, val_loader, criterion1, criterion2, 
                                                                    alpha, val_dice_metric, device, post_pred)
            
            # Log validation results
            log_results(writer, epoch, val_loss, val_dice_scores, "Val")
            log_sensitivity(writer, epoch, sensitivity)
            
            # Save best models
            save_best_models(model, optimizer, epoch, val_loss, val_dice_scores, sensitivity, 
                             best_LI, best_LI_sens, best_loss, checkpoints_dir)
        
        # Save last model
        save_model(model, optimizer, epoch, val_loss, os.path.join(checkpoints_dir, "last_model.pth"))
        
        print()  # Add a blank line for readability in console output

def initialize_metrics():
    """Initialize and return Dice metrics for training and validation."""
    train_dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    val_dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    return train_dice_metric, val_dice_metric

def train_epoch(model, train_loader, optimizer, criterion1, criterion2, alpha, train_dice_metric, post_pred, accumulation_steps, device):
    """Perform one epoch of training."""
    model.train()
    train_loss = 0.0
    train_dice_metric.reset()
    
    for b_index, batch in enumerate(tqdm(train_loader, desc=f"Training")):
        inputs, targets = batch['image'].to(device), batch['label'].to(device)
        targets = preprocess_targets(targets)
        
        outputs = model(inputs)
        train_dice_metric(y_pred=torch.stack([post_pred(output) for output in outputs]), y=targets)
        
        loss = calculate_loss(outputs, targets, criterion1, criterion2, alpha)
        loss.backward()
        
        if (b_index + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item()
    
    if (b_index + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(train_loader)
    train_dice_scores = train_dice_metric.aggregate()
    
    return train_loss, train_dice_scores

def validate_model(model, val_loader, criterion1, criterion2, alpha, val_dice_metric, device, post_pred):
    """Perform validation on the model."""
    model.eval()
    val_loss = 0.0
    val_dice_metric.reset()
    sensitivity_metric = ConfusionMatrixMetric(include_background=False, metric_name="sensitivity", 
                                               compute_sample=True, reduction="mean_batch")
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            targets = preprocess_targets(targets)
            
            predictions = predict_full_volume(model, inputs, patch_size=PATCH_SIZE, stride=STRIDE)
            
            val_dice_metric(y_pred=torch.stack([post_pred(output) for output in predictions]), y=targets)
            sensitivity_metric(y_pred=torch.stack([post_pred(output) for output in predictions]), y=targets)
            
            loss = calculate_loss(predictions, targets, criterion1, criterion2, alpha)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_dice_scores = val_dice_metric.aggregate()
    sensitivity = sensitivity_metric.aggregate()
    
    return val_loss, val_dice_scores, sensitivity

def predict_full_volume(model, inputs, patch_size, stride):
    """Predict on full volume using sliding window approach."""
    B, C, D, H, W = inputs.shape
    predictions = torch.zeros_like(inputs)
    count = torch.zeros_like(inputs)
    
    for d in range(0, D - patch_size + 1, stride):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                patch = inputs[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                prediction = model(patch)
                predictions[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] += prediction
                count[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1
    
    return predictions / count

def preprocess_targets(targets):
    """Preprocess target labels."""
    targets = torch.cat([(targets.sum(dim=1) == 0).unsqueeze(1), targets], dim=1)
    targets[:, 1, :, :, :] = targets[:, 1, :, :, :] + targets[:, 7, :, :, :]
    return targets[:, :7, :, :, :]

def calculate_loss(outputs, targets, criterion1, criterion2, alpha):
    """Calculate the combined loss."""
    if alpha == 1:
        return criterion1(outputs, targets)
    elif alpha == 0:
        return criterion2(outputs, targets)
    else:
        return 2 * (alpha * criterion1(outputs, targets) + (1-alpha) * criterion2(outputs, targets))

def log_results(writer, epoch, loss, dice_scores, phase):
    """Log results to TensorBoard."""
    writer.add_scalar(f'Loss/{phase}', loss, global_step=epoch)
    for class_idx, score in enumerate(dice_scores[0]):
        writer.add_scalar(f'Dice/{phase}/Class_{class_idx}', score.item(), global_step=epoch)

def log_sensitivity(writer, epoch, sensitivity):
    """Log sensitivity scores to TensorBoard."""
    for class_idx, score in enumerate(sensitivity[0]):
        writer.add_scalar(f'Sensitivity/Class_{class_idx}', score.item(), global_step=epoch)

def save_best_models(model, optimizer, epoch, val_loss, val_dice_scores, sensitivity, best_LI, best_LI_sens, best_loss, checkpoints_dir):
    """Save the best models based on different metrics."""
    if val_dice_scores[0][3] > best_LI:
        best_LI = val_dice_scores[0][3].item()
        save_model(model, optimizer, epoch, val_loss, os.path.join(checkpoints_dir, "bestLI.pth"))
    
    if sensitivity[0][3] > best_LI_sens:
        best_LI_sens = sensitivity[0][3].item()
        save_model(model, optimizer, epoch, val_loss, os.path.join(checkpoints_dir, "bestLIsens.pth"))
    
    if val_loss < best_loss:
        best_loss = val_loss
        save_model(model, optimizer, epoch, val_loss, os.path.join(checkpoints_dir, "best_model.pth"))

def save_model(model, optimizer, epoch, val_loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, path)