import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

import wandb
from Unet.dataset import BinarizationDataset
from Unet.model import AttentionUNet
from Unet.losses import DiceLoss, FocalLoss, DiceBCELoss, FocalDiceIoUMSELoss, dice_coeff, iou_coeff

# --- Directories ---
DIR_BASE = Path('./')
DIR_DATA = DIR_BASE / 'Train_Dataset_512'
DIR_CHECKPOINTS = Path('./model_checkpoints/')

CPU_NUM = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- THIS IS THE UPDATED EVALUATION FUNCTION ---
def evaluate(model, dataloader, device, loss_fn):
    """Evaluation loop for the model on the validation set."""
    model.eval()
    num_val_batches = len(dataloader)
    
    # Reset metrics
    val_loss = 0
    dice_score = 0
    iou_score = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            # Predict the mask
            mask_pred_logits = model(images)
            
            # Calculate loss (using the same loss function as training, e.g., SamLoss)
            val_loss += loss_fn(mask_pred_logits, true_masks).item()
            
            # Calculate metrics
            dice_score += dice_coeff(mask_pred_logits, true_masks)
            iou_score += iou_coeff(mask_pred_logits, true_masks)

    model.train() # Set the model back to training mode
    
    if num_val_batches == 0:
        return 0, 0, 0
        
    # Return average loss and metrics
    avg_loss = val_loss / num_val_batches
    avg_dice = dice_score / num_val_batches
    avg_iou = iou_score / num_val_batches
    
    return avg_loss, avg_dice, avg_iou


def save_validation_images(model, dataloader, device, epoch, save_dir):
    """Saves a grid of validation images: input, ground truth, and prediction."""
    model.eval()
    # Get one batch from the validation dataloader
    try:
        images, true_masks = next(iter(dataloader))
    except StopIteration:
        logging.warning("Validation dataloader is empty. Cannot save images.")
        return None

    images = images.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_logits = model(images)
        pred_probs = torch.sigmoid(pred_logits)
        pred_masks = (pred_probs > 0.5).float()

    true_masks_rgb = true_masks.repeat(1, 3, 1, 1)
    pred_masks_rgb = pred_masks.repeat(1, 3, 1, 1)

    # Concatenate images side-by-side for comparison
    comparison_grid = torch.cat([images.cpu(), true_masks_rgb.cpu(), pred_masks_rgb.cpu()], dim=3)
    
    grid = make_grid(comparison_grid, nrow=4, normalize=True)
    
    # Save the grid
    epoch_save_dir = save_dir / f'epoch_{epoch:03d}'
    epoch_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = epoch_save_dir / 'validation_previews.png'
    save_image(grid, save_path)
    
    logging.info(f"Saved validation images to {save_path}")
    model.train()
    return grid


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        data_dir: str = str(DIR_DATA),
        loss_function: str = 'bce',
        use_wandb: bool = True
):
    # 1. Create dataset
    train_dataset = BinarizationDataset(
        DIR_DATA / 'train' / 'images',
        DIR_DATA / 'train' / 'masks',
        transform=None
    )
    
    val_dataset = BinarizationDataset(
        DIR_DATA / 'val' / 'images',
        DIR_DATA / 'val' / 'masks',
        transform=None
    )

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=CPU_NUM, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    
    # 4. Initialize logging (wandb) - only if enabled
    experiment = None
    if use_wandb:
        run_name = f"gated_ICVGIP_Final_512_{wandb.util.generate_id()}"
        experiment = wandb.init(
            project='Unet-Binarization-ICVGIP',
            name=run_name,
            resume='allow',
            anonymous='must'
        )
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      val_percent=val_percent, save_checkpoint=save_checkpoint,
                                      amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Wandb logging:   {use_wandb}
    ''')

    # 5. Set up the optimizer, loss, scheduler and AMP scaler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    
    # Choose the loss function based on user input
    print(f"Using loss function: {loss_function}")
    if loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == 'dice':
        criterion = DiceLoss()
    elif loss_function == 'focal':
        criterion = FocalLoss()
    elif loss_function == 'dice_bce':
        criterion = DiceBCELoss()
    elif loss_function == 'focal_dice':
        focal_loss_fn = FocalLoss() 
        dice_loss_fn = DiceLoss()
        alpha = 0.5
        criterion = lambda pred, target: alpha * focal_loss_fn(pred, target) + (1 - alpha) * dice_loss_fn(pred, target)
    elif loss_function == 'sam':
        criterion = FocalDiceIoUMSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    global_step = 0
    best_dice = 0.0

    # 6. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch
                
                assert images.shape[1] == model.in_channels, \
                    f'Network has been defined with {model.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                # Log to wandb only if enabled
                if use_wandb and experiment:
                    experiment.log({
                        'train_loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        val_score_loss, val_dice, val_iou = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_dice)

        # Save validation preview images
        val_grid = save_validation_images(model, val_loader, device, epoch, DIR_CHECKPOINTS)

        logging.info(f'Validation Loss: {val_score_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        
        # Log metrics to wandb only if enabled
        if use_wandb and experiment:
            experiment.log({
                'learning_rate': optimizer.param_groups[0]['lr'],
                'validation_loss': val_score_loss,
                'validation_dice': val_dice,
                'validation_iou': val_iou,
                'epoch': epoch,
                'step': global_step,
                'validation_images': wandb.Image(val_grid) if val_grid is not None else None
            })

        # Save checkpoint if it's the best one yet
        if save_checkpoint:
            state_dict = model.state_dict()
            torch.save(state_dict, str(DIR_CHECKPOINTS / f'last_model.pth'))
            if val_dice > best_dice:
                best_dice = val_dice
                epoch_save_dir = DIR_CHECKPOINTS / f'epoch_{epoch:03d}'
                epoch_save_dir.mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(epoch_save_dir / f'best_model_dice_{best_dice:.4f}.pth'))
                logging.info(f'Checkpoint {epoch} saved! Best Dice score: {best_dice:.4f}')
    
    # Finish wandb run if it was initialized
    if use_wandb and experiment:
        wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--data-dir', type=str, default=str(DIR_DATA),
                        help='Base directory for the data')
    parser.add_argument('--val-percent', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--loss', type=str, default='dice_bce',
                        help='Loss function to use. Options: "bce", "dice", "focal", "dice_bce", "sam')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable Weights & Biases logging')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = AttentionUNet(in_channels=3, out_classes=1, bilinear=args.bilinear)
    model.to(device=device)
    
    logging.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{model.out_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            amp=args.amp,
            data_dir=args.data_dir,
            loss_function=args.loss,
            use_wandb=args.wandb
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)