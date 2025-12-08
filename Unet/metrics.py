import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Dice coefficient for a batch of predictions. This is a "hard" Dice,
    meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The Dice coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate the intersection and the sums of the masks
    intersection = (inputs_flat * targets_flat).sum()
    dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

    return dice_score


def iou_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Intersection over Union (IoU) coefficient, also known as the Jaccard index.
    This is a "hard" IoU, meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The IoU coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate intersection and union
    intersection = (inputs_flat * targets_flat).sum()
    total = (inputs_flat + targets_flat).sum()
    union = total - intersection

    # Calculate the IoU score
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score

def dice_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Dice coefficient for a batch of predictions. This is a "hard" Dice,
    meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The Dice coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate the intersection and the sums of the masks
    intersection = (inputs_flat * targets_flat).sum()
    dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

    return dice_score


def iou_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Intersection over Union (IoU) coefficient, also known as the Jaccard index.
    This is a "hard" IoU, meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The IoU coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate intersection and union
    intersection = (inputs_flat * targets_flat).sum()
    total = (inputs_flat + targets_flat).sum()
    union = total - intersection

    # Calculate the IoU score
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score


def bwmorph_thin(image, n_iter=None):
    """A Python port of Matlab's bwmorph(I, 'thin', n)."""
    X = image.astype(bool)
    if n_iter is None:
        n_iter = -1  # Run until convergence

    for i in range(n_iter) if n_iter != -1 else iter(int, 1):
        X_prev = X.copy()
        # Step 1
        C = (~X) & (X[1:, :] | np.vstack((np.zeros((1, X.shape[1]), dtype=bool), ~X[:-1, :])))
        N = X[:-1, :] | np.vstack((X[1:, :], np.zeros((1, X.shape[1]), dtype=bool)))
        S = X[1:, :] | np.vstack((np.zeros((1, X.shape[1]), dtype=bool), X[:-1, :]))
        W = X[:, 1:] | np.hstack((np.zeros((X.shape[0], 1), dtype=bool), X[:, :-1]))
        E = X[:, :-1] | np.hstack((X[:, 1:], np.zeros((X.shape[0], 1), dtype=bool)))
        C1 = N & E & (S | W) & ~X
        X[C1] = False
        # Step 2
        N = X[:-1, :] | np.vstack((X[1:, :], np.zeros((1, X.shape[1]), dtype=bool)))
        S = X[1:, :] | np.vstack((np.zeros((1, X.shape[1]), dtype=bool), X[:-1, :]))
        W = X[:, 1:] | np.hstack((np.zeros((X.shape[0], 1), dtype=bool), X[:, :-1]))
        E = X[:, :-1] | np.hstack((X[:, 1:], np.zeros((X.shape[0], 1), dtype=bool)))
        C2 = S & W & (N | E) & ~X
        X[C2] = False
        if np.all(X == X_prev):
            break
    return X.astype(np.uint8)


def drd_fn(im, im_gt):
    """Calculates the Distance Reciprocal Distortion (DRD) metric."""
    try:
        height, width = im.shape
        neg = np.zeros(im.shape, dtype=np.uint8)
        neg[im_gt != im] = 1
        
        n = 2
        m = n * 2 + 1
        W = np.zeros((m, m), dtype=np.uint8)
        W[n, n] = 1
        W = cv2.distanceTransform(1 - W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        W[n, n] = 1.
        W = 1. / W
        W[n, n] = 0.
        W /= W.sum()

        nubn = 0.
        block_size = 8
        for y1 in range(0, height, block_size):
            for x1 in range(0, width, block_size):
                y2 = min(y1 + block_size, height)
                x2 = min(x1 + block_size, width)
                block_dim = (y2 - y1) * (x2 - x1)
                block = 1 - im_gt[y1:y2, x1:x2]
                block_sum = np.sum(block)
                if 0 < block_sum < block_dim:
                    nubn += 1
        
        if nubn == 0:
            return 0.0

        drd_sum = 0.
        y_coords, x_coords = np.where(neg == 1)

        for i in range(len(y_coords)):
            y, x = y_coords[i], x_coords[i]
            x1 = max(0, x - n)
            y1 = max(0, y - n)
            x2 = min(width - 1, x + n)
            y2 = min(height - 1, y + n)
            local_gt = im_gt[y1:y2+1, x1:x2+1]
            local_W_x1, local_W_y1 = n - (x - x1), n - (y - y1)
            local_W_x2, local_W_y2 = local_W_x1 + (x2 - x1) + 1, local_W_y1 + (y2 - y1) + 1
            local_W = W[local_W_y1:local_W_y2, local_W_x1:local_W_x2]
            drd_sum += np.sum(np.abs(im[y, x] - local_gt) * local_W)

        return drd_sum / nubn
    except Exception:
        return np.nan


def calculate_binarization_metrics(im_pred, im_gt):
    """Calculates a suite of binarization metrics from NumPy arrays."""
    im_pred[im_pred > 0] = 1
    im_gt[im_gt > 0] = 1
    
    height, width = im_pred.shape
    npixel = height * width

    tp = np.sum((im_pred == 0) & (im_gt == 0))
    tn = np.sum((im_pred == 1) & (im_gt == 1))
    fp = np.sum((im_pred == 0) & (im_gt == 1))
    fn = np.sum((im_pred == 1) & (im_gt == 0))

    try:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fmeasure = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    except ZeroDivisionError: fmeasure = 0.0

    try:
        sk_gt = bwmorph_thin(im_gt == 0)
        precall = np.sum((im_pred == 0) & sk_gt) / np.sum(sk_gt) if np.sum(sk_gt) > 0 else 0
        pfmeasure = (2 * precall * precision) / (precall + precision) if (precall + precision) > 0 else 0
    except ZeroDivisionError: pfmeasure = 0.0

    try:
        mse = (fp + fn) / npixel
        psnr = 10. * np.log10(1. / mse) if mse > 0 else 100.0
    except ZeroDivisionError: psnr = 100.0

    try:
        nrfn = fn / (fn + tp) if (fn + tp) > 0 else 0
        nrfp = fp / (fp + tn) if (fp + tn) > 0 else 0
        nrm = (nrfn + nrfp) / 2
    except ZeroDivisionError: nrm = 0.0

    try:
        kernel = np.ones((3, 3), dtype=np.uint8)
        im_gt_border = cv2.dilate(im_gt, kernel) - im_gt
        dist = cv2.distanceTransform(1 - im_gt_border, cv2.DIST_L2, 3)
        nd = np.sum(dist)
        mpfn = np.sum(dist[(im_pred == 1) & (im_gt == 0)]) / nd if nd > 0 else 0
        mpfp = np.sum(dist[(im_pred == 0) & (im_gt == 1)]) / nd if nd > 0 else 0
        mpm = (mpfp + mpfn) / 2
    except ZeroDivisionError: mpm = 0.0
    
    drd = drd_fn(im_pred, im_gt)

    return {
        'F-Measure': fmeasure,
        'pF-Measure': pfmeasure,
        'PSNR': psnr,
        'NRM': nrm,
        'MPM': mpm,
        'DRD': drd
    }