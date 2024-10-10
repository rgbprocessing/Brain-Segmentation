# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:48:46 2024

@author: Dragon
"""

def create_incremented_run_folder(base_path="./runs"):
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
    run_dir = create_incremented_run_folder(base_path="./runs_patches")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs", "tensorboard_logs")
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return run_dir, checkpoints_dir, logs_dir

def train_model(model, train_loader, val_loader, criterion1, criterion2, optimizer, num_epochs, device):
    # Initialize Dice metric for training and validation
    train_dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    val_dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    best_loss = 100.00
    alpha = .5
    post_pred = AsDiscrete(argmax=True, to_onehot=7)  # 7 is the number of classes
    accumulation_steps = 1
    optimizer.zero_grad()
    best_LI = 0
    best_LI_sens = 0
    best_LI_trn = 0
    


    sensitivity_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        compute_sample=True,
        reduction="mean_batch"
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice_metric.reset()
        
        for b_index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, targets = batch['image'].to(device), batch['label'].to(device)
            # Add the new class channel
            targets = torch.cat([(targets.sum(dim=1) == 0).unsqueeze(1), targets], dim=1)
            targets[:,1,:,:,:]=targets[:,1,:,:,:]+targets[:,7,:,:,:]
            targets = targets[:,:7,:,:,:]
            
            
            #optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate Dice scores
            train_dice_metric(y_pred=torch.stack([post_pred(output) for output in outputs]), y=targets)
            
            if alpha==1:
                loss = criterion1(outputs, targets)
            elif alpha==0:
                loss = criterion2(outputs, targets)
            else:
                loss = 2* (alpha * criterion1(outputs, targets) + (1-alpha) * criterion2(outputs, targets))
                
            loss.backward()
            #optimizer.step()
            if (b_index + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            
        if (b_index + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= len(train_loader)
        train_dice_scores = train_dice_metric.aggregate()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar(f'Loss/Train', train_loss, global_step=epoch)
        print("Train Dice Scores:")
        for class_idx, score in enumerate(train_dice_scores[0]):
            print(f"Class {class_idx + 1}: {score.item():.4f}")
            writer.add_scalar(f'Dice/Train/Class_{class_idx}', score.item(), global_step=epoch)
        
        val_loss = 0.0
        if train_dice_scores[0][3]>.06: #(best_LI_trn-(best_LI_trn*.1))
            best_LI_trn = train_dice_scores[0][3].item()
        
            # Validation
            model.eval()
            
            val_dice_metric.reset()
            sensitivity_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch['image'].to(device), batch['label'].to(device)
                    patch_size=32
                    stride=16
                    # Add background glass to labels
                    targets = torch.cat([(targets.sum(dim=1) == 0).unsqueeze(1), targets], dim=1)
                    targets[:,1,:,:,:]=targets[:,1,:,:,:]+targets[:,7,:,:,:]
                    targets = targets[:,:7,:,:,:]
                    
                    B, C, D, H, W = targets.shape
                
                    # Create an empty tensor to store the predictions
                    ##predictions = torch.zeros_like(targets)
                    ##count = torch.zeros_like(targets)
                    
                    predictions = torch.zeros_like(targets)
                    count = torch.zeros_like(targets)
                    
                    # Loop over the volume with the given stride
                    for d in range(0, D - patch_size + 1, stride):
                        for h in range(0, H - patch_size + 1, stride):
                            for w in range(0, W - patch_size + 1, stride):
                                # Extract patch
                                patch = inputs[:,:, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                                
                                # Get prediction
                                prediction = model(patch)
                                
                                
                                # Add prediction to the full volume
                                predictions[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] += prediction
                                count[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size] += 1
                    
                    final_prediction = predictions / count
                    val_dice_metric(y_pred=torch.stack([post_pred(output) for output in final_prediction]), y=targets)
                    sensitivity_metric(y_pred=torch.stack([post_pred(output) for output in final_prediction]),  y=targets)
    
                    if alpha==1:
                        loss = criterion1(final_prediction, targets)
                    elif alpha==0:
                        loss = criterion2(final_prediction, targets)
                    else:
                        loss = 2* (alpha * criterion1(final_prediction, targets) + (1-alpha) * criterion2(final_prediction, targets))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_dice_scores = val_dice_metric.aggregate()
            sensitivity = sensitivity_metric.aggregate()
        
        
        
            print("Validation Dice Scores:")
            writer.add_scalar(f'Loss/Val', val_loss, global_step=epoch)
            print(f"Val Loss: {val_loss:.4f}")
            for class_idx, score in enumerate(val_dice_scores[0]):
                print(f"Class {class_idx + 1}: {score.item():.4f}")
                writer.add_scalar(f'Dice/Val/Class_{class_idx}', score.item(), global_step=epoch)
            for class_idx, score in enumerate(sensitivity[0]):
                print(f"Class {class_idx + 1}: {score.item():.4f}")
                writer.add_scalar(f'Dice/Sensitivity/Class_{class_idx}', score.item(), global_step=epoch)
        
            # Here you can add code to save the model, update learning rate, etc.
            if val_dice_scores[0][3]>best_LI:
                best_LI = val_dice_scores[0][3].item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(checkpoints_dir, "besLI.pth"))
            if sensitivity[0][3]>best_LI_sens:
                best_LI_sens = sensitivity[0][3].item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(checkpoints_dir, "besLIsens.pth"))
            if val_loss<best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(checkpoints_dir, "best_model.pth"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, os.path.join(checkpoints_dir, "last_model.pth"))
        
        print()