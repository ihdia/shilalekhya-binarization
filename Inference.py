import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import json
import random

from Unet.model import AttentionUNet
from Unet.fast_dibco_metrics import compute_dibco_metrics

# ==============================================================================
# 1. HELPER FUNCTIONS 
# ==============================================================================

def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image)

def resize_and_pad(image, target_size, is_mask=False):
    h, w = image.shape[:2]
    if max(h, w) <= target_size:
        return image
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)

def get_range_color(h_multiplier, h_range):
    range_size = h_range[1] - h_range[0]
    step = range_size / 3.0
    if h_multiplier <= h_range[0] + step: return (255, 0, 0)
    elif h_multiplier <= h_range[0] + 2 * step: return (0, 255, 0)
    else: return (0, 0, 255)

def _calculate_iqr_filtered_mean(data: np.ndarray) -> float:
    if data.size == 0: return 0.0
    q1, q3 = np.percentile(data, [30, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inliers = data[(data >= lower_bound) & (data <= upper_bound)]
    return float(np.mean(inliers)) if inliers.size > 0 else float(np.median(data))

def get_robust_component_stats(prediction: np.ndarray):
    pred_bin = (prediction > 0.5).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    if num_labels <= 1:
        return {'iqr_mean_height': 0.0}
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_indices = areas >= 10
    heights = heights[valid_indices]
    if heights.size == 0:
        return {'iqr_mean_height': 0.0}
    return {'iqr_mean_height': _calculate_iqr_filtered_mean(heights)}

def predict_full_image_patchwise(model, image_tensor, device, patch_size, overlap, in_size):
    model.eval()
    c, h, w = image_tensor.shape
    stride = patch_size - overlap
    pad_h = (stride - (h - overlap) % stride) % stride
    pad_w = (stride - (w - overlap) % stride) % stride
    image_padded = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect').unsqueeze(0)
    _, _, H, W = image_padded.shape
    prediction_canvas = torch.zeros((1, 1, H, W), device=device)
    count_canvas = torch.zeros((1, 1, H, W), device=device)
    for y in tqdm(range(0, H - overlap, stride), desc=f"Patch Size {patch_size}", leave=False):
        for x in range(0, W - overlap, stride):
            patch = image_padded[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                patch_in = F.interpolate(patch, size=(in_size, in_size), mode='bilinear', align_corners=False)
                pred_patch = torch.sigmoid(model(patch_in.to(device)))
                pred_patch_orig_size = F.interpolate(pred_patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            prediction_canvas[:, :, y:y+patch_size, x:x+patch_size] += pred_patch_orig_size
            count_canvas[:, :, y:y+patch_size, x:x+patch_size] += 1
    prediction_canvas /= (count_canvas + 1e-8)
    return prediction_canvas[:, :, :h, :w].squeeze().cpu().numpy()

def fuse_predictions(masks, method='max'):
    if not masks: raise ValueError("Mask list cannot be empty.")
    if method == 'average': return np.mean(np.stack(masks), axis=0)
    elif method == 'max': return np.maximum.reduce(masks)
    else: raise ValueError(f"Unknown fusion method: {method}")

def get_dynamic_patch_definitions(pseudo_gt_mask, base_char_h, h_range=(3, 9), patch_params=None):
    if patch_params is None: patch_params = {'fg_patches': 250, 'bg_patches': 75}
    h_img, w_img = pseudo_gt_mask.shape
    guidance_bin = (pseudo_gt_mask > 0.5).astype(np.uint8)
    fg_pixels = np.argwhere(guidance_bin > 0)
    bg_pixels = np.argwhere(guidance_bin == 0)
    if len(fg_pixels) == 0: return []
    num_fg = min(patch_params['fg_patches'], len(fg_pixels))
    num_bg = min(patch_params['bg_patches'], len(bg_pixels))
    fg_indices = np.random.choice(len(fg_pixels), num_fg, replace=False)
    patch_defs = [{'center': c, 'h_multiplier': random.uniform(h_range[0], h_range[1])} for c in fg_pixels[fg_indices]]
    if num_bg > 0:
        bg_indices = np.random.choice(len(bg_pixels), num_bg, replace=False)
        patch_defs.extend([{'center': c, 'h_multiplier': random.uniform(h_range[0], h_range[1])} for c in bg_pixels[bg_indices]])
    random.shuffle(patch_defs)
    print(f"Generated {len(patch_defs)} dynamic patch definitions ({num_fg} FG, {num_bg} BG).")
    return patch_defs

def save_comparison_visualization(image_tensor, gt_mask, stage1_pred, stage2_pred, 
                               save_path, stage1_metrics, final_metrics, image_name):
    """Creates and saves a side-by-side comparison with metrics in filename."""
    # Convert tensors to numpy arrays and normalize
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_np = (gt_mask.squeeze().numpy() * 255).astype(np.uint8)
    stage1_np = (stage1_pred * 255).astype(np.uint8)
    stage2_np = (stage2_pred).astype(np.uint8)  # Assuming already in range [0, 255]
    
    # Convert single channel to 3-channel
    gt_rgb = cv2.cvtColor(gt_np, cv2.COLOR_GRAY2BGR)
    stage1_rgb = cv2.cvtColor(stage1_np, cv2.COLOR_GRAY2BGR)
    stage2_rgb = cv2.cvtColor(stage2_np, cv2.COLOR_GRAY2BGR)
    
    # Get max height
    max_h = max(image_np.shape[0], gt_rgb.shape[0], 
                stage1_rgb.shape[0], stage2_rgb.shape[0])
    
    # Resize all images to same height while maintaining aspect ratio
    def resize_maintain_aspect(img, target_h):
        h, w = img.shape[:2]
        target_w = int(w * (target_h / h))
        return cv2.resize(img, (target_w, target_h))
    
    # save org img in a new folder
    save_path = Path(save_path)
    new_path = save_path / 'original_images'
    new_path.mkdir(parents=True, exist_ok=True)
    original_image_path = new_path / f"{image_name}.png"
    cv2.imwrite(str(original_image_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    image_resized = resize_maintain_aspect(image_np, max_h)
    gt_resized = resize_maintain_aspect(gt_rgb, max_h)
    stage1_resized = resize_maintain_aspect(stage1_rgb, max_h)
    stage2_resized = resize_maintain_aspect(stage2_rgb, max_h)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    # Concatenate horizontally
    comparison = np.hstack([image_resized, gt_resized, stage1_resized, stage2_resized])
    
    # Add metrics to filename
    metrics_str = f"{image_name}_s1_fm{stage1_metrics['fmeasure']:.3f}_drd{stage1_metrics['drd']:.1f}_s2_fm{final_metrics['fmeasure']:.3f}_drd{final_metrics['drd']:.1f}.png"
    base_path = Path(save_path)
    save_name = f"{metrics_str}{base_path.suffix}"
    print(str(base_path / save_name))
    cv2.imwrite(str(base_path / save_name), comparison)
    
    return comparison

# ==============================================================================
# 2. MAIN INFERENCE ORCHESTRATOR
# ==============================================================================

def run_hybrid_inference(model, img_path, device, save_dirs, config):
    """Orchestrates the hybrid two-stage inference process."""
    # --- Basic Setup ---
    image_name = Path(img_path).stem
    print(f"\n===== Processing {image_name} with Hybrid Fusion + Dynamic Method =====")
    original_image_tensor = load_and_preprocess_image(img_path)
    image_np = resize_and_pad(original_image_tensor.permute(1,2,0).numpy(), config['RESIZE_SIZE'])
    image_tensor = torch.from_numpy(image_np).permute(2,0,1)
    _, h, w = image_tensor.shape

    # --- STAGE 1: Generate Pseudo-GT via Multi-Scale Fusion ---
    print("\n--- Stage 1: Generating High-Quality Pseudo-GT via Multi-Scale Fusion ---")
    prediction_masks_stage1 = []
    stage1_stats = {}  # Track stats for each scale
    
    for patch_size in config['FUSION_SCALES']:
        print(f"\nProcessing scale: {patch_size}x{patch_size}")
        current_patch_size = min(patch_size, h, w)
        pred_mask = predict_full_image_patchwise(
            model, image_tensor, device, current_patch_size, current_patch_size // 2, config['IN_SIZE']
        )
        prediction_masks_stage1.append(pred_mask)
        
        # Save the prediction mask for this scale
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
        save_path = os.path.join(save_dirs['pseudo_gt'], f'{image_name}_scale_{patch_size}_pred.png')
        cv2.imwrite(save_path, pred_mask_bin)
        
        # Compute and store statistics for this scale
        stats = get_robust_component_stats(pred_mask)
        stage1_stats[patch_size] = {
            'char_height': stats['iqr_mean_height'],
            'components': len(np.unique(cv2.connectedComponents(
                (pred_mask > 0.5).astype(np.uint8))[1])) - 1,
            'fg_ratio': np.mean(pred_mask > 0.5)
        }
    
    # Print Stage 1 Statistics
    print("\n=== Stage 1 Statistics ===")
    print(f"{'Scale':<10} {'Char Height':>12} {'Components':>12} {'FG Ratio':>10}")
    print("-" * 44)
    max_scale = 0
    max_h = 0
    for scale, stats in stage1_stats.items():
        if max_h <= int(stats['char_height']):
            max_scale = scale
            max_h = int(stats['char_height'])
        print(f"{scale:<10} {stats['char_height']:>12.2f} {stats['components']:>12d} {stats['fg_ratio']:>10.2%}")
    
    pseudo_gt_mask_float = fuse_predictions(prediction_masks_stage1, method=config['FUSION_METHOD'])
    
    # Print fused result statistics
    fused_stats = get_robust_component_stats(pseudo_gt_mask_float)
    fused_components = len(np.unique(cv2.connectedComponents(
        (pseudo_gt_mask_float > 0.5).astype(np.uint8))[1])) - 1
    print("\n=== Fused Result Statistics ===")
    print(f"Fusion Method    : {config['FUSION_METHOD']}")
    print(f"Character Height : {fused_stats['iqr_mean_height']:.2f}px")
    print(f"Components      : {fused_components}")
    print(f"FG Ratio       : {np.mean(pseudo_gt_mask_float > 0.5):.2%}")
    
    # Save the superior Pseudo-GT for inspection
    pseudo_gt_bin = (pseudo_gt_mask_float > 0.5).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(save_dirs['pseudo_gt'], f'{image_name}_fused_pseudo_gt.png'), pseudo_gt_bin)

    # After saving pseudo_gt_bin, add DIBCO metrics computation for stage 1
    print("\n=== Stage 1 DIBCO Metrics ===")
    mask_path = os.path.join(config['MASK_DIR'], f"{image_name}.png")
    stage1_metrics = None
    if os.path.exists(mask_path):
        mask_np = cv2.resize(np.array(Image.open(mask_path).convert('L')), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
        pseudo_gt_tensor = torch.from_numpy((pseudo_gt_bin / 255.0).astype(np.float32)).unsqueeze(0)
        stage1_metrics = compute_dibco_metrics(pseudo_gt_tensor, mask_tensor)
        for key, value in stage1_metrics.items():
            print(f"{key:<12}: {value:.4f}")
    
    # --- Bridge: Analyze Pseudo-GT to guide Stage 2 ---
    stats = get_robust_component_stats(pseudo_gt_mask_float)
    base_char_h = stats['iqr_mean_height']
    if base_char_h < 15: # Sanity check for failed Pseudo-GT
        print(f"Warning: Low char height ({base_char_h:.2f}px) from Pseudo-GT. Falling back to default of 45px.")
        base_char_h = 50.0
    else:
        print(f"Estimated base character height from Pseudo-GT: {base_char_h:.2f}px")
    
    if (max_scale / 9) > base_char_h:
        base_char_h = max_scale // 9
    elif max_h > base_char_h:
        base_char_h = max_h
    print(f'base_h {base_char_h}, max_scale/9 {max_scale // 9}')

    # --- STAGE 2: Refine with Dynamic Patching ---
    print("\n--- Stage 2: Refining prediction with targeted Dynamic Patching ---")
    patch_defs = get_dynamic_patch_definitions(pseudo_gt_mask_float, base_char_h, config['H_RANGE'], config['PATCH_PARAMS'])

    if not patch_defs:
        print("Dynamic patching failed. Using Fused Pseudo-GT as final result.")
        final_pred_bin = pseudo_gt_bin
    else:
        # Calculate safe padding size
        max_h_multiplier = config['H_RANGE'][1]
        max_patch_size = int(max_h_multiplier * base_char_h)
        pad_size = min(h - 1, w - 1)  # Limit pad size to 1/4 of image dimensions
        
        print(f"Using pad size: {pad_size} (original max patch size: {max_patch_size})")
        
        # Pad the original image for patch extraction
        image_padded_tensor = F.pad(image_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        _, H_pad, W_pad = image_padded_tensor.shape
        
        max_patch_size = min(max_patch_size, H_pad, W_pad)
        print(f"Using pad size: {pad_size} (original max patch size: {max_patch_size})")
        
        # Initialize canvases and visualization
        prediction_canvas = torch.zeros((1, H_pad, W_pad), device='cpu')
        count_canvas = torch.zeros((1, H_pad, W_pad), device='cpu', dtype=torch.int16)
        viz_img = cv2.cvtColor(image_padded_tensor.permute(1,2,0).cpu().numpy()*255, cv2.COLOR_RGB2BGR).astype(np.uint8).copy()

        for p_def in tqdm(patch_defs, desc="Stage 2 Dynamic Inference"):
            center_y, center_x = p_def['center']
            h_multiplier = p_def['h_multiplier']
            padded_center_y, padded_center_x = center_y + pad_size, center_x + pad_size
            dynamic_patch_size = max(48, int(h_multiplier * base_char_h))
            dynamic_patch_size = min(dynamic_patch_size, pad_size)
            half_patch = dynamic_patch_size // 2
            y1, y2 = padded_center_y - half_patch, padded_center_y + half_patch
            x1, x2 = padded_center_x - half_patch, padded_center_x + half_patch

            patch = image_padded_tensor[:, y1:y2, x1:x2].unsqueeze(0)
            with torch.no_grad():
                patch_in = F.interpolate(patch, size=(config['IN_SIZE'], config['IN_SIZE']), mode='bilinear', align_corners=False)
                pred_prob = torch.sigmoid(model(patch_in.to(device)))
                pred_prob_orig_size = F.interpolate(pred_prob.cpu(), size=(y2-y1, x2-x1), mode='bilinear', align_corners=False)
            prediction_canvas[:, y1:y2, x1:x2] += pred_prob_orig_size.squeeze(0)
            count_canvas[:, y1:y2, x1:x2] += 1
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), get_range_color(h_multiplier, config['H_RANGE']), 3)

        # Finalize stitching and un-pad
        cv2.imwrite(os.path.join(save_dirs['visualizations'], f'{image_name}_stage2_viz.jpg'), viz_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        prediction_canvas /= (count_canvas + 1e-8)
        final_prediction_float_unpadded = prediction_canvas.squeeze().numpy()[pad_size:pad_size+h, pad_size:pad_size+w]
        final_pred_bin = (final_prediction_float_unpadded > 0.5).astype(np.uint8) * 255
    
    # --- Finalization and Evaluation ---
    cv2.imwrite(os.path.join(save_dirs['predictions'], f'{image_name}_final_hybrid_pred.png'), final_pred_bin)
    
    metrics = None
    mask_path = os.path.join(config['MASK_DIR'], f"{image_name}.png")
    if os.path.exists(mask_path):
        mask_np = cv2.resize(np.array(Image.open(mask_path).convert('L')), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
        final_pred_tensor = torch.from_numpy((final_pred_bin / 255.0).astype(np.float32)).unsqueeze(0)
        metrics = compute_dibco_metrics(final_pred_tensor, mask_tensor)
        metrics['base_height'] = base_char_h
        print("\n=== Final Hybrid Method DIBCO Metrics ===")
        for key, value in metrics.items(): print(f"{key:<12}: {value:.4f}")
        
        # Save comparison visualization
        save_comparison_visualization(
            image_tensor,
            mask_tensor,
            pseudo_gt_mask_float,
            final_pred_bin,
            save_dirs['comparisons'],
            stage1_metrics,
            metrics,
            image_name
        )
            
    return stage1_metrics, metrics

# ==============================================================================
# 3. MAIN SCRIPT EXECUTION
# ==============================================================================

def main():
    # --- Configuration ---
    input_dir = ''  # Directory containing images
    
    config = {
        # --- PATHS ---
        'MASK_DIR': '', # Directory containing ground truth masks for the images
        'MODEL_PATH': '',
        'OUTPUT_DIR': '',
        
        # --- GENERAL PARAMS ---
        'IN_SIZE': 512,
        'RESIZE_SIZE': 1536,
        
        # --- STAGE 1: FUSION CONFIG ---
        'FUSION_SCALES': [256, 384, 512, 784],
        'FUSION_METHOD': 'max',
        
        # --- STAGE 2: DYNAMIC PATCHING CONFIG ---
        'H_RANGE': (4, 9),
        'PATCH_PARAMS': {'fg_patches': 90, 'bg_patches': 20}
    }

    # --- Setup Output Directories ---
    save_dirs = {
        'predictions': os.path.join(config['OUTPUT_DIR'], 'final_predictions'),
        'pseudo_gt': os.path.join(config['OUTPUT_DIR'], 'pseudo_gt_stage1'),
        'visualizations': os.path.join(config['OUTPUT_DIR'], 'visualizations_stage2'),
        'summary': os.path.join(config['OUTPUT_DIR'], 'summary'),
        'comparisons': os.path.join(config['OUTPUT_DIR'], 'comparisons')
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # --- Load Model ---
    print("Loading model...")
    model = AttentionUNet(in_channels=3, out_classes=1, bilinear=False)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(config['MODEL_PATH'], map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Get list of images to process ---
    supported_formats = ['.png', '.jpg', '.jpeg', '.tiff']
    image_files = [f for f in os.listdir(input_dir) 
                  if any(f.lower().endswith(fmt) for fmt in supported_formats)]
    
    if not image_files:
        print(f"No supported images found in {input_dir}")
        return

    # --- Process all images ---
    all_results = {}
    stage1_metrics_all = []
    final_metrics_all = []

    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {img_file}")
            print(f"{'='*50}")
            
            stage1_metrics, final_metrics = run_hybrid_inference(
                model, img_path, device, save_dirs, config
            )
            
            # Store results
            result_data = {
                'method': 'Hybrid (Fusion -> Dynamic)',
                'fusion_method_stage1': config['FUSION_METHOD'],
                'fusion_scales_stage1': config['FUSION_SCALES'],
                'stage1_metrics': {k: float(v) for k, v in stage1_metrics.items()} if stage1_metrics else "No GT provided",
                'final_metrics': {k: float(v) for k, v in final_metrics.items()} if final_metrics else "No GT provided"
            }
            all_results[img_file] = result_data
            
            # Collect metrics for averaging
            if isinstance(stage1_metrics, dict):
                stage1_metrics_all.append(stage1_metrics)
            if isinstance(final_metrics, dict):
                final_metrics_all.append(final_metrics)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            all_results[img_file] = {'error': str(e)}

    # --- Save Complete Summary ---
    summary_path = os.path.join(save_dirs['summary'], 'complete_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # --- Print Final Summary ---
    print("\n\n========== COMPLETE PROCESSING SUMMARY ==========")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully processed: {len(final_metrics_all)}")
    print(f"Failed processes: {len(image_files) - len(final_metrics_all)}")
    
    # Calculate and print average metrics
    if stage1_metrics_all and final_metrics_all:
        print("\nAverage Stage 1 (Fusion) Metrics:")
        avg_metrics_stage1 = {}
        for metric in stage1_metrics_all[0].keys():
            values = [m[metric] for m in stage1_metrics_all]
            avg_metrics_stage1[metric] = sum(values) / len(values)
            print(f"{metric:<12}: {avg_metrics_stage1[metric]:.4f}")
            
        print("\nAverage Final (Hybrid) Metrics:")
        avg_metrics_final = {}
        for metric in final_metrics_all[0].keys():
            values = [m[metric] for m in final_metrics_all]
            avg_metrics_final[metric] = sum(values) / len(values)
            print(f"{metric:<12}: {avg_metrics_final[metric]:.4f}")
            
        # Save average metrics
        avg_summary_path = os.path.join(save_dirs['summary'], 'average_metrics.json')
        with open(avg_summary_path, 'w') as f:
            json.dump({
                'total_images': len(image_files),
                'successful_processes': len(final_metrics_all),
                'average_stage1_metrics': avg_metrics_stage1,
                'average_final_metrics': avg_metrics_final
            }, f, indent=4)
    
    print(f"\nComplete summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
    print("\nAll done!")