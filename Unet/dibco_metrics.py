import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Tuple
import random
# from lightning.pytorch import LightningModule
# import lightning as L
import os
import cv2
import math
from scipy import ndimage as ndi

# Add DIBCO metrics lookup tables and functions
G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1,
       0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0], dtype=bool)

def bwmorph_thin(image, n_iter=None):
    """Morphological thinning of a binary image"""
    if n_iter is None:
        n = -1
    elif n_iter <= 0:
        raise ValueError('n_iter must be > 0')
    else:
        n = n_iter
    
    skel = np.array(image).astype(np.uint8)
    
    if skel.ndim != 2:
        raise ValueError('2D array required')
    if not np.all(np.in1d(image.flat,(0,1))):
        raise ValueError('Image contains values other than 0 and 1')

    mask = np.array([[ 8,  4,  2],
                     [16,  0,  1],
                     [32, 64,128]],dtype=np.uint8)

    while n != 0:
        before = np.sum(skel)
        
        for lut in [G123_LUT, G123P_LUT]:
            N = ndi.correlate(skel, mask, mode='constant')
            D = np.take(lut, N)
            skel[D] = 0
            
        after = np.sum(skel)
        
        if before == after:
            break
        n -= 1
    
    return skel.astype(bool)

def drd_fn(im, im_gt):
    """Distance Reciprocal Distortion calculation"""
    height, width = im.shape
    neg = np.zeros(im.shape)
    neg[im_gt!=im] = 1
    y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
    
    n = 2
    m = n*2+1
    W = np.zeros((m,m), dtype=np.uint8)
    W[n,n] = 1.
    W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    W[n,n] = 1.
    W = 1./W
    W[n,n] = 0.
    W /= W.sum()
    
    nubn = 0.
    block_size = 8
    for y1 in range(0, height, block_size):
        for x1 in range(0, width, block_size):
            y2 = min(y1+block_size-1, height-1)
            x2 = min(x1+block_size-1, width-1)
            block_dim = (x2-x1+1)*(y2-y1+1)
            block = 1-im_gt[y1:y2+1, x1:x2+1]
            block_sum = np.sum(block)
            if block_sum>0 and block_sum<block_dim:
                nubn += 1
    
    if nubn == 0:
        return 0.0
    
    drd_sum = 0.
    tmp = np.zeros(W.shape)
    
    for i in range(len(y)):
        tmp[:,:] = 0 

        x1 = max(0, x[i]-n)
        y1 = max(0, y[i]-n)
        x2 = min(width-1, x[i]+n)
        y2 = min(height-1, y[i]+n)

        yy1 = y1-y[i]+n
        yy2 = y2-y[i]+n
        xx1 = x1-x[i]+n
        xx2 = x2-x[i]+n

        tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
        tmp *= W

        drd_sum += np.sum(tmp)
    
    return drd_sum/nubn

def psnr_metric(img1, img2):
    """PSNR calculation"""
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_dibco_metrics(pred_tensor, target_tensor):
    """
    Compute DIBCO-style metrics from PyTorch tensors
    Args:
        pred_tensor: Predicted probabilities [1, H, W] 
        target_tensor: Ground truth binary mask [1, H, W] with values in {0, 1}
    Returns:
        dict with fmeasure, pfmeasure, psnr, drd
    """
    # Convert to numpy and threshold
    if pred_tensor.dim() == 3:
        pred_np = pred_tensor[0].cpu().numpy()
    else:
        pred_np = pred_tensor.cpu().numpy()
        
    if target_tensor.dim() == 3:
        target_np = target_tensor[0].cpu().numpy()
    else:
        target_np = target_tensor.cpu().numpy()
    
    # Threshold prediction to binary
    _, pred_binary = cv2.threshold(pred_np, 0.5, 1, cv2.THRESH_BINARY)
    _, target_binary = cv2.threshold(target_np, 0.5, 1, cv2.THRESH_BINARY)
    
    height, width = pred_binary.shape
    npixel = height * width
    
    # Ensure binary values
    pred_binary[pred_binary > 0] = 1
    target_binary[target_binary > 0] = 1
    
    # Skeleton calculation for pseudo F-measure
    try:
        sk = bwmorph_thin(1 - target_binary)
        im_sk = np.ones(target_binary.shape)
        im_sk[sk] = 0
        
        # Pseudo TP (skeleton-based)
        ptp = np.zeros(target_binary.shape)
        ptp[(pred_binary == 0) & (im_sk == 0)] = 1
        numptp = ptp.sum()
    except:
        # Fallback if skeleton calculation fails
        numptp = 0
        im_sk = np.ones(target_binary.shape)
    
    # Standard TP, TN, FP, FN
    tp = np.zeros(target_binary.shape)
    tp[(pred_binary == 0) & (target_binary == 0)] = 1
    numtp = tp.sum()
    if numtp == 0:    
        numtp = 1

    tn = np.zeros(target_binary.shape)
    tn[(pred_binary == 1) & (target_binary == 1)] = 1
    numtn = tn.sum()    
    
    fp = np.zeros(target_binary.shape)
    fp[(pred_binary == 0) & (target_binary == 1)] = 1
    numfp = fp.sum()    
    
    fn = np.zeros(target_binary.shape)
    fn[(pred_binary == 1) & (target_binary == 0)] = 1
    numfn = fn.sum()    

    # Calculate precision and recall with zero checks
    if (numtp + numfp + numfn) == 0:
        precision = 1.0
        recall = 1.0
    else:
        if (numtp + numfp) == 0:
            precision = 0.0
        else:
            precision = numtp / (numtp + numfp)
        
        if (numtp + numfn) == 0:
            recall = 0.0
        else:
            recall = numtp / (numtp + numfn)
    
    # Calculate pseudo recall (precall)
    skeleton_sum = np.sum(1 - im_sk)
    if skeleton_sum == 0:
        precall = 0.0
    else:
        precall = numptp / skeleton_sum
    
    # Calculate F-measure
    if (recall + precision) == 0:
        fmeasure = 0.0
    else:
        fmeasure = (2 * recall * precision) / (recall + precision)

    # Calculate pseudo F-measure
    if (precall + precision) == 0:
        pfmeasure = 0.0
    else:
        pfmeasure = (2 * precall * precision) / (precall + precision)   

    # Calculate DICE coefficient (same as F1-score for binary classification)
    if numtp == 0:
        dice = 0.0
    else:
        dice = (2 * numtp) / (2 * numtp + numfp + numfn)
    
    # Calculate IoU (Intersection over Union)
    if numtp == 0:
        iou = 0.0
    else:
        iou = numtp / (numtp + numfp + numfn)

    # Calculate PSNR
    mse = (numfp + numfn) / npixel
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10. * np.log10(1. / mse)
    
    # Alternative PSNR calculation
    psnr2 = psnr_metric(pred_binary, target_binary)
    
    # Calculate DRD
    try:
        drd = drd_fn(pred_binary, target_binary)
    except:
        drd = 0.0
    
    return {
        'fmeasure': fmeasure,
        'pfmeasure': pfmeasure, 
        'psnr': psnr,
        'psnr2': psnr2,
        'drd': drd,
        'precision': precision,
        'recall': recall,
        'precall': precall,
        'dice': dice,
        'iou': iou
    }
