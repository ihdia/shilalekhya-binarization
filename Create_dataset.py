import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import json
import datetime

F_PATCH = 0
B_PATCH = 0

# ==============================================================================
# Helper functions for robust statistics and dynamic patch counting
# ==============================================================================
def _calculate_iqr_filtered_mean(data: np.ndarray) -> float:
    if data.size == 0: return 0.0
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inliers = data[(data >= lower_bound) & (data <= upper_bound)]
    return float(np.mean(inliers)) if inliers.size > 0 else float(np.median(data))

def get_robust_component_stats(gt_mask: np.ndarray):
    pred_bin = (gt_mask > 128).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    if num_labels <= 1: return {'iqr_mean_height': 0.0}
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area_threshold = 5
    valid_indices = areas >= min_area_threshold
    heights = heights[valid_indices]
    if heights.size == 0: return {'iqr_mean_height': 0.0}
    return {'iqr_mean_height': _calculate_iqr_filtered_mean(heights)}

# NEW function to determine patch count
def get_content_based_patch_count(gt_mask: np.ndarray, base_rate: int, min_p: int, max_p: int) -> int:
    pred_bin = (gt_mask > 128).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    if num_labels <= 1: return min_p
        
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if heights.size == 0: return min_p
        
    q1_h, q3_h = np.percentile(heights, [25, 75])
    iqr_h = q3_h - q1_h
    lower_bound_h = q1_h - 1.5 * iqr_h
    upper_bound_h = q3_h + 1.5 * iqr_h
    
    min_area_threshold = 5
    valid_indices = (heights >= lower_bound_h) & (heights <= upper_bound_h) & (areas >= min_area_threshold)
    num_valid_components = np.sum(valid_indices)

    suggested_patches = int((num_valid_components / 10.0) * base_rate)
    return max(min_p, min(max_p, suggested_patches))

def get_background_patch_count(gt_mask: np.ndarray, valid_area_mask: np.ndarray, mean_height: int,max_bg_patches: int) -> int:
    if gt_mask is None or gt_mask.size == 0:
        return 0

    # The total area is now the number of valid pixels, not the whole image rectangle.
    total_effective_area = np.sum(valid_area_mask)
    if total_effective_area == 0:
        return 0

    pred_bin = (gt_mask > 128).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)

    if num_labels <= 1:
        # If no text, the whole valid area is background.
        return max_bg_patches

    # Create the canvas for drawing the union of bounding boxes.
    fg_bbox_canvas = np.zeros_like(gt_mask, dtype=np.uint8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_comp = stats[i, cv2.CC_STAT_WIDTH]
        h_comp = stats[i, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(fg_bbox_canvas, (x, y), (x + w_comp, y + h_comp), color=255, thickness=cv2.FILLED)
    
    # kernel_size = 5
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # fg_bbox_canvas = cv2.dilate(fg_bbox_canvas, kernel, iterations=3)
    
    # Create rectangular kernel with wider horizontal dimension
    kernel_h = max(3, int(mean_height * 0.3))  # 30% of mean height
    kernel_w = kernel_h * 3  # Make kernel 3x wider than tall
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    
    # Apply horizontal dilation first to connect characters
    fg_bbox_canvas = cv2.dilate(fg_bbox_canvas, kernel, iterations=2)
    
    # Then apply a small vertical dilation for safety margin
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    fg_bbox_canvas = cv2.dilate(fg_bbox_canvas, kernel_vertical, iterations=1)

    # Calculate the foreground area ONLY within the valid region.
    # This is the intersection of the bounding box union and the valid area mask.
    semantic_fg_area = np.sum((fg_bbox_canvas > 0) & valid_area_mask)
    
    # The background area is what's left of the *effective* area.
    semantic_bg_area = total_effective_area - semantic_fg_area

    # Calculate the ratio based on the effective area.
    bg_ratio = semantic_bg_area / total_effective_area if total_effective_area > 0 else 0
    # print(bg_ratio, num_labels)
    # suggested_patches = min(int(bg_ratio * max_bg_patches), int(bg_ratio*num_labels))
    suggested_patches = int(bg_ratio * max_bg_patches)
    
    # suggested_patches = int(suggested_patches * 0.7)
    
    fg_bbox_canvas[~valid_area_mask] = 128

    return suggested_patches, fg_bbox_canvas

# Add this helper function at the top with other helper functions
def get_range_color(h_multiplier, h_range):
    """
    Returns a color based on where h_multiplier falls in h_range
    Returns BGR color tuple
    """
    # Define colors for different ranges (in BGR)
    range_size = h_range[1] - h_range[0]
    step = range_size / 3
    
    if h_multiplier <= h_range[0] + step:
        return (255, 0, 0)  # Blue for small patches
    elif h_multiplier <= h_range[0] + 2*step:
        return (0, 255, 0)  # Green for medium patches
    else:
        return (0, 0, 255)  # Red for large patches

# ==============================================================================
# Main Dynamic Patch Creation Logic
# ==============================================================================
def create_dynamic_patches_for_image(img_path, gt_path, output_dir, output_size, h_range, patch_count_params):
    try:
        # Load the image WITH its alpha channel, if it exists
        image_with_alpha = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if image_with_alpha is None or gt_mask is None: return 0
    except Exception as e: return 0
    
    # Check for and fix dimensional mismatches between image and mask
    h_mask, w_mask = gt_mask.shape[:2]
    h_img, w_img = image_with_alpha.shape[:2]

    if h_mask != h_img or w_mask != w_img:
        # Use tqdm.write to avoid breaking the progress bar display
        tqdm.write(f"Warning: Mismatch for {img_path.name}. "
                   f"Image: {(h_img, w_img)}, Mask: {(h_mask, w_mask)}. "
                   f"Resizing image to match mask dimensions.")
        
        # Resize the image to match the mask's dimensions.
        # cv2.resize expects (width, height)
        image_with_alpha = cv2.resize(image_with_alpha, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR)

    # --- NEW: Alpha Channel Handling ---
    if image_with_alpha.ndim == 3 and image_with_alpha.shape[2] == 4:
        # Image has an alpha channel. Create a validity mask from it.
        alpha_channel = image_with_alpha[:, :, 3]
        valid_area_mask = (alpha_channel > 10) # Use a small threshold for robustness
        # Convert image back to 3-channel BGR for processing
        image = cv2.cvtColor(image_with_alpha, cv2.COLOR_BGRA2BGR)
    else:
        # Image is JPG or does not have an alpha channel. The whole area is valid.
        valid_area_mask = np.ones(gt_mask.shape, dtype=bool)
        image = image_with_alpha # It's already a 3-channel or grayscale image

    # 1. Calculate base character height
    stats = get_robust_component_stats(gt_mask)
    base_char_h = stats['iqr_mean_height']
    if base_char_h < 5: return 0

    # 2. NEW: Determine how many patches to create for this image
    num_patches_to_create = get_content_based_patch_count(
        gt_mask,
        base_rate=patch_count_params['base_rate'],
        min_p=patch_count_params['min_patches'],
        max_p=patch_count_params['max_patches'],
    )
    
    # 2b. Determine BACKGROUND patch count based on background area
    num_bg_patches, bg_img = get_background_patch_count(
        gt_mask,
        valid_area_mask,
        base_char_h,
        max_bg_patches=patch_count_params['max_bg_patches'] # New parameter
    )
    
    global F_PATCH
    global B_PATCH
    F_PATCH += num_patches_to_create
    B_PATCH += num_bg_patches
    
    ## pad the image
    pad_size = int(h_range[1]) * int(stats['iqr_mean_height'])
    image_padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    # gt_padded = cv2.copyMakeBorder(gt_mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
    gt_padded = cv2.copyMakeBorder(gt_mask, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    # 3. Find all possible centers for patches
    foreground_pixels = np.argwhere((bg_img > 250) & valid_area_mask)
    background_pixels = np.argwhere((bg_img <= 0) & valid_area_mask)
    if len(foreground_pixels) == 0: return 0
    
    viz_img = image_padded.copy()
    gt_1 = cv2.cvtColor(gt_padded.copy(), cv2.COLOR_GRAY2BGR)
    gt_2 = cv2.cvtColor(gt_padded.copy(), cv2.COLOR_GRAY2BGR)
        
    # 4. Extract Patches (logic is mostly the same, just uses the new patch count)
    patches_created = 0
    img_h, img_w = image.shape[:2]
    
    # We will sample num_patches_to_create unique centers
    if len(foreground_pixels) < num_patches_to_create:
        # If not enough unique pixels, sample with replacement
        selected_indices = np.random.choice(len(foreground_pixels), num_patches_to_create, replace=True)
        selected_indices_bg = np.random.choice(len(background_pixels), int(num_bg_patches), replace=True)
    else:
        selected_indices = random.sample(range(len(foreground_pixels)), num_patches_to_create)
        selected_indices_bg = random.sample(range(len(background_pixels)), int(num_bg_patches))

    # Combine foreground and background centers
    selected_centers_fg = foreground_pixels[selected_indices]
    selected_centers_bg = background_pixels[selected_indices_bg]
    selected_centers = np.vstack((selected_centers_fg, selected_centers_bg))
    
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    # Draw foreground points in green
    for center_y, center_x in selected_centers_fg:
        cv2.circle(bg_img, (center_x, center_y), 3, (0, 255, 0), -1)  # Green for foreground

    # Draw background points in red
    for center_y, center_x in selected_centers_bg:
        cv2.circle(bg_img, (center_x, center_y), 3, (0, 0, 255), -1)
    
    bg_patches_created = 0
    for i, (center_y, center_x) in enumerate(selected_centers):
        h_multiplier = random.uniform(h_range[0], h_range[1])
        # print(h_multiplier)
        dynamic_patch_size = int(h_multiplier * base_char_h)
        dynamic_patch_size = max(48, min(dynamic_patch_size, img_h, img_w))
        # print(dynamic_patch_size)

        y1 = max(0, center_y - dynamic_patch_size // 2)
        x1 = max(0, center_x - dynamic_patch_size // 2)
        y2 = min(img_h, y1 + dynamic_patch_size)
        x2 = min(img_w, x1 + dynamic_patch_size)
        
        # Adjust center coordinates for the padded image
        padded_center_y, padded_center_x = center_y + pad_size, center_x + pad_size

        # Extract the patch of DYNAMIC size from the PADDED image
        half_patch = dynamic_patch_size // 2
        y1, y2 = padded_center_y - half_patch, padded_center_y + half_patch
        x1, x2 = padded_center_x - half_patch, padded_center_x + half_patch
        
        if y2-y1 < 32 or x2-x1 < 32: continue

        img_patch = image_padded[y1:y2, x1:x2]
        gt_patch = gt_padded[y1:y2, x1:x2]
        
        # Color the patches based on h_multiplier
        text_pos = (x1, y1-5)
        color = get_range_color(h_multiplier, h_range)
        if patches_created <= num_patches_to_create:
            # if patches_created <=24:
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(gt_1, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(viz_img, f'{h_multiplier:.1f}', text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # Background patches in gray
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (64,64,64), 2)
            # if bg_patches_created <=8:
            cv2.rectangle(gt_2, (x1, y1), (x2, y2), (0,0,255), 2)
            bg_patches_created +=1
            cv2.putText(viz_img, f'{h_multiplier:.1f}', text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1)
        
        # Add text showing h_multiplier
        # text_pos = (x1, y1-5)
        # cv2.putText(viz_img, f'{h_multiplier:.1f}', text_pos, 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        img_patch_resized = cv2.resize(img_patch, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        gt_patch_resized = cv2.resize(gt_patch, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        
        base_name = img_path.stem
        img_patch_path = output_dir / "images" / f"{base_name}_patch_{i}.png"
        gt_patch_path = output_dir / "masks" / f"{base_name}_patch_{i}.png"
        cv2.imwrite(str(img_patch_path), img_patch_resized)
        cv2.imwrite(str(gt_patch_path), gt_patch_resized)
        patches_created += 1
        
        
    # Save visualization image
    os.makedirs(str(output_dir / "visualizations"), exist_ok=True)
    vis_path = output_dir / "visualizations" / f"{img_path.stem}_patches.png"
    cv2.imwrite(str(vis_path), viz_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    
    # vis_path = output_dir / "visualizations" / f"{img_path.stem}_org_pad.png"
    # cv2.imwrite(str(vis_path), image_padded, [cv2.IMWRITE_JPEG_QUALITY, 75])

    ## Save background area visualization
    
    # bg_vis_path = output_dir / "visualizations" / f"{img_path.stem}_bg_areas.png"
    # cv2.imwrite(str(bg_vis_path), bg_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    # bg_vis_path = output_dir / "visualizations" / f"{img_path.stem}_bg_gt1.png"
    # cv2.imwrite(str(bg_vis_path), gt_1, [cv2.IMWRITE_JPEG_QUALITY, 75])
    # bg_vis_path = output_dir / "visualizations" / f"{img_path.stem}_bg_gt2.png"
    # cv2.imwrite(str(bg_vis_path), gt_2, [cv2.IMWRITE_JPEG_QUALITY, 75])
    # bg_vis_path = output_dir / "visualizations" / f"{img_path.stem}_bg_mask.png"
    # cv2.imwrite(str(bg_vis_path), gt_mask, [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    print(f"Created {num_patches_to_create} foreground and {num_bg_patches} background patches")
        
    return patches_created

def main(args):
    # Create train/val/test directories
    for split in ['train', 'val', 'test']:
        (Path(args.output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / split / "masks").mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / split / "visualizations").mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir)
    
    # Load train/val/test split
    split_file = './difficulty_split_easy_v1.json'
    try:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
            # Remove patch counts from image names if present
            # split_data = {k: [img.split('_')[0] for img in v] for k, v in split_data.items()}
        print(f"Loaded split data with {len(split_data['train'])} train, {len(split_data['val'])} val, {len(split_data['test'])} test images")
    except Exception as e:
        print(f"Could not load split data: {e}")
        return

    patch_count_params = {
        'base_rate': args.base_rate,
        'min_patches': args.min_patches,
        'max_patches': args.max_patches,
        'max_bg_patches': args.max_bg_patches
    }

    total_patches = {split: 0 for split in ['train', 'val', 'test']}
    patches_num = {split: {} for split in ['train', 'val', 'test']}

    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        image_list = split_data[split]
        
        for img_name in tqdm(image_list, desc=f"Creating {split} patches"):
            img_path = image_dir / f"{img_name}.png"
            if not img_path.exists():
                img_path = image_dir / f"{img_name}.jpg"
                if not img_path.exists(): 
                    continue

            gt_path = gt_dir / f"{img_name}_mask.png"
            if not gt_path.exists():
                gt_path = gt_dir / f"{img_name}_mask.jpg"
                if not gt_path.exists(): 
                    continue

            patches_created = create_dynamic_patches_for_image(
                img_path, gt_path, 
                Path(args.output_dir) / split,  # Save to split-specific directory
                args.output_size, args.h_range, patch_count_params
            )
            
            if patches_created > 0:
                tqdm.write(f"Created {patches_created} patches for {img_path.name}")
                patches_num[split][img_name] = patches_created
                total_patches[split] += patches_created

    # Print summary
    print("\nPatch creation complete:")
    for split in ['train', 'val', 'test']:
        print(f"{split}: {total_patches[split]} patches")

    # Save patches_num dictionary to a JSON file
    patches_info_path = Path(args.output_dir) / 'patches_info.json'
    with open(patches_info_path, 'w') as f:
        json.dump(patches_num, f, indent=4)
    
    print(f"Patches information saved to {patches_info_path}")

if __name__ == "__main__":
    # Define simple folder paths
    IMAGE_FOLDER = "./binarized_masks"
    GT_FOLDER = "./binarized_masks"
    OUTPUT_FOLDER = "./Train_Dataset_512"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    parser = argparse.ArgumentParser(description="Create fully content-aware dynamic patches for training.")
    
    parser.add_argument("--image_dir", type=str, default=IMAGE_FOLDER, help="Directory containing original images.")
    parser.add_argument("--gt_dir", type=str, default=GT_FOLDER, help="Directory containing ground truth masks.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help="Directory to save output patches.")
    
    parser.add_argument("--output_size", type=int, default=512, help="Final pixel dimension of saved patches.")
    parser.add_argument("--h_range", type=int, nargs=2, default=[4, 12], help="Range of character height multipliers [min, max].")
    
    # New arguments for dynamic patch counting
    parser.add_argument("--base_rate", type=int, default=5, help="Number of patches to generate per 10 valid characters.")
    parser.add_argument("--min_patches", type=int, default=10, help="Minimum number of patches to generate per image.")
    parser.add_argument("--max_patches", type=int, default=250, help="Maximum number of patches to generate per image.")
    
    # New argument for background sampling
    parser.add_argument("--max_bg_patches", type=int, default=75,
                        help="Maximum number of background patches to sample, scaled by BG area. An image with 90%% background will get 0.9*50=45 patches.")

    args = parser.parse_args()
    
    config = {
        "data_paths": {
            "image_dir": str(args.image_dir),
            "gt_dir": str(args.gt_dir),
            "output_dir": str(args.output_dir)
        },
        "patch_params": {
            "output_size": args.output_size,
            "height_multiplier_range": args.h_range,
            "base_rate": args.base_rate,
            "min_patches": args.min_patches,
            "max_patches": args.max_patches,
            "max_bg_patches": args.max_bg_patches
        },
        "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save config to JSON
    config_path = Path(args.output_dir) / 'dataset_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    
    main(args)