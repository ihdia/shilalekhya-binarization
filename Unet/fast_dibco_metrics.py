import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import math
from scipy import ndimage as ndi
from skimage.morphology import thin  

# --- Precompute constants ---
# Build distance-reciprocal weight kernel for DRD

def build_weight_kernel(n: int = 2) -> np.ndarray:
    m = 2 * n + 1
    # binary mask: center pixel = 1, others = 0
    center = np.zeros((m, m), dtype=np.uint8)
    center[n, n] = 1
    # L2 distance transform
    dist = cv2.distanceTransform(1 - center, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # avoid division by zero at center
    dist[n, n] = 1.0
    W = 1.0 / dist
    W[n, n] = 0.0
    # normalize
    W /= W.sum()
    return W

W_DRD = build_weight_kernel(n=2)

# --- Modified DRD using vectorized convolution and block reshape ---

def drd_fn_fast(im: np.ndarray, im_gt: np.ndarray) -> float:
    """
    Vectorized Distance Reciprocal Distortion (DRD)
    """
    # boolean error mask
    err_mask = (im_gt != im).astype(np.uint8)

    # weighted error via convolution
    drd_map = cv2.filter2D(err_mask, ddepth=-1, kernel=W_DRD, borderType=cv2.BORDER_CONSTANT)
    drd_sum = drd_map.sum()

    # count non-uniform 8x8 blocks
    h, w = im_gt.shape
    H8, W8 = h // 8 * 8, w // 8 * 8
    trim_gt = im_gt[:H8, :W8]
    blocks = trim_gt.reshape(H8 // 8, 8, W8 // 8, 8)
    sums = blocks.sum(axis=(1, 3))
    nubn = np.count_nonzero((sums > 0) & (sums < 8*8))

    return float(drd_sum / nubn) if nubn > 0 else 0.0

# --- Main metric computation ---
def compute_dibco_metrics(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> dict:
    """
    Compute DIBCO-style metrics from PyTorch tensors with black background (0) and white foreground (1)
    Returns dict with fmeasure, pfmeasure, psnr, drd, precision, recall, dice, iou
    """
    # Convert tensors to numpy arrays
    pred_np = pred_tensor.squeeze().cpu().numpy()
    target_np = target_tensor.squeeze().cpu().numpy()

    # Convert to 8-bit [0,255] for reliable thresholding
    pred_8 = (pred_np * 255).astype(np.uint8)
    tgt_8 = (target_np * 255).astype(np.uint8)

    # Binary threshold at midpoint - now 1 is white (foreground)
    _, pred_binary = cv2.threshold(pred_8, 127, 1, cv2.THRESH_BINARY)
    _, target_binary = cv2.threshold(tgt_8, 127, 1, cv2.THRESH_BINARY)

    # ---- Pseudo F-measure (use C-optimized thinning) ----
    try:
        # Now we thin the foreground directly (no need for 1 - target_binary)
        skel = thin(target_binary.astype(np.uint8))
        im_sk = np.zeros_like(target_binary)  # Initialize with black background
        im_sk[skel == 1] = 1  # Set skeleton to white

        # Pseudo true positive - now checking for white pixels (1)
        ptp = ((pred_binary == 1) & (im_sk == 1)).astype(np.uint8)
        numptp = ptp.sum()
    except Exception:
        numptp = 0
        im_sk = np.zeros_like(target_binary)

    # ---- Standard confusion counts - now 1 is foreground ----
    tp = ((pred_binary == 1) & (target_binary == 1)).sum() or 1
    fp = ((pred_binary == 1) & (target_binary == 0)).sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum()
    tn = ((pred_binary == 0) & (target_binary == 0)).sum()

    # Precision & Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Pseudo recall
    skeleton_sum = im_sk.sum()
    precall = numptp / skeleton_sum if skeleton_sum > 0 else 0.0

    # F-measures
    fmeasure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    pfmeasure = (2 * precision * precall) / (precision + precall) if (precision + precall) > 0 else 0.0

    # Dice & IoU
    dice = (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp > 0 else 0.0

    # PSNR 
    npixel = pred_binary.size
    mse = (fp + fn) / npixel
    psnr = 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)

    # DRD via fast function
    drd = drd_fn_fast(pred_binary, target_binary)

    return {
        'fmeasure': float(fmeasure),
        'pfmeasure': float(pfmeasure),
        'precision': float(precision),
        'recall': float(recall),
        'dice': float(dice),
        'iou': float(iou),
        'psnr': float(psnr),
        'drd': float(drd)
    }

def compute_dibco_metrics_simple(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> dict:
    """
    Compute simplified DIBCO metrics (dice, iou, fmeasure) from PyTorch tensors.
    Assumes black background (0) and white foreground (1).
    Returns dict with dice, iou, fmeasure
    """
    # Convert tensors to numpy arrays
    pred_np = pred_tensor.squeeze().cpu().numpy()
    target_np = target_tensor.squeeze().cpu().numpy()

    # Convert to binary
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    target_binary = (target_np > 0.5).astype(np.uint8)

    # Calculate confusion matrix elements
    tp = ((pred_binary == 1) & (target_binary == 1)).sum() or 1
    fp = ((pred_binary == 1) & (target_binary == 0)).sum()
    fn = ((pred_binary == 0) & (target_binary == 1)).sum()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F-measure
    fmeasure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate Dice & IoU
    dice = (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp > 0 else 0.0

    return {
        'fmeasure': float(fmeasure),
        'dice': float(dice),
        'iou': float(iou)
    }

# --- Parallelized worker function ---
import concurrent.futures
import os

def _worker_fn(args):
    """Helper function for multiprocessing. Takes numpy arrays, returns metrics dict."""
    pred_np, target_np = args
    # Convert numpy arrays back to tensors for the original function
    pred_tensor = torch.from_numpy(pred_np)
    target_tensor = torch.from_numpy(target_np)
    return compute_dibco_metrics(pred_tensor, target_tensor)


def compute_dibco_metrics_batch_parallel(pred_tensor: torch.Tensor, target_tensor: torch.Tensor, num_workers: int = None) -> dict:
    """
    Computes DIBCO-style metrics for a batch of images in parallel and returns the average.
    This function uses a process pool to distribute calculations across multiple CPU cores.

    Args:
        pred_tensor (torch.Tensor): Batch of predicted images, shape (B, C, H, W).
                                    Values should be in [0, 1].
        target_tensor (torch.Tensor): Batch of ground truth images, shape (B, C, H, W).
                                      Values should be in [0, 1].
        num_workers (int, optional): The number of CPU cores to use. 
                                     If None, it defaults to the number of CPUs on the machine.
                                     Defaults to None.

    Returns:
        dict: A dictionary containing the average of each metric over the batch.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    if pred_tensor.dim() == 3: pred_tensor = pred_tensor.unsqueeze(0)
    if target_tensor.dim() == 3: target_tensor = target_tensor.unsqueeze(0)

    batch_size = pred_tensor.shape[0]
    if batch_size == 0:
        metric_keys = ['fmeasure', 'pfmeasure', 'precision', 'recall', 'dice', 'iou', 'psnr', 'drd']
        return {key: 0.0 for key in metric_keys}

    # Convert tensors to numpy arrays in the main process ONCE
    # This is more efficient than sending tensors, which might need to be pickled
    pred_np_batch = pred_tensor.cpu().numpy()
    target_np_batch = target_tensor.cpu().numpy()

    # Create a list of arguments for the worker function
    tasks = [(pred_np_batch[i], target_np_batch[i]) for i in range(batch_size)]

    batch_metrics_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map() applies the worker function to each task and returns results as they complete
        results_iterator = executor.map(_worker_fn, tasks)
        batch_metrics_list = list(results_iterator)

    # Aggregate the results by averaging
    aggregated_metrics = {key: sum(d[key] for d in batch_metrics_list) for key in batch_metrics_list[0]}
    avg_metrics = {key: value / batch_size for key, value in aggregated_metrics.items()}

    return avg_metrics