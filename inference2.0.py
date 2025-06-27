'''
# Inference script for the MusselFinder project

# This script performs semantic segmentation on images using a pre-trained Segformer model.
# It processes images in tiles, calculates coverage, and saves the results to a CSV file.
# It also generates overlay images with contours and saves them to the output directory.
# It includes options for resuming interrupted inference, plotting overlays, and adjusting image scaling.

# Copyright (C) 2023-2024 Angus Galloway (
# Copyright (C) 2023-2025 Anthony Geglio (

'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import csv
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import PIL
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from my_transformer_utils import SemanticSegmentationDatasetDisplay
# from src.plot_utils import draw_pink_contours
from utils import Utils
from plot_utils import draw_pink_contours

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        help="mandatory path to model checkpoints",
        default=r"Z:\__AdvancedTechnologyBackup\04_ProjectData\Proj_MusselFinder\musselfinder-CheckPoints\exp-b (Elk Rapids)-20240806T153527Z-001\exp-b (Elk Rapids)\pretrained\segformer_b3_dot1_512px\checkpoint-4062"
    )
    parser.add_argument("--image_dir", help="root path", default=None)
    parser.add_argument("--scalar", help="scaling for pixel size correction, 0.1 to 2.0", default=1.0, type=float)
    parser.add_argument("--cam_sys", help="camera system used", default="Abiss2", choices=['Abiss1', 'Abiss2', 'REMUS'], type=str)
    parser.add_argument("--output_dir", help="output image path", default="output")
    parser.add_argument("--output_name", help="output folder name", default="inference_results")
    parser.add_argument("--resume", help="resume if inference was interrupted", action="store_true")
    parser.add_argument("--plot", help="plot contours overlays", action="store_true")
    parser.add_argument("--plot_every", help="plot every x image overlays", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size for inference dataloader", type=int, default=1)
    parser.add_argument("--tile_prop", type=float, default=0.3, help="Proportion of image to use for tile size (e.g. 0.3 means 30%% of image height/width)")
    return parser.parse_args()


def create_csv_file(out_csv):
    """ Create a CSV file to store inference results.
    If the file already exists, return the list of filenames already processed.
    """
    try:
        with open(out_csv, "x", newline='') as f:
            csv.writer(f).writerow(["Filename", "Coverage", "infer_imh", "infer_imw", "infer_scalar", "n_tiles"])
            return []
    except FileExistsError:
        with open(out_csv, "r") as f:
            return [row.split(",")[0] for row in f]

def initialize_model(ckpt, device):
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt)
    model.to(device)
    model.eval()
    return model

def sliding_window_inference(image_tensor, model, device, tile_size, stride):
    """
    Perform sliding window inference with overlap and blend results.
    image_tensor: (1, 3, H, W)
    Returns: (H, W) prediction mask
    """
    _, _, H, W = image_tensor.shape
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride

    # Prepare output arrays
    # Assuming model.config.num_labels is correct for the number of classes
    output_probs = torch.zeros((model.config.num_labels, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Robust tile position calculation
    def get_positions(img_dim, tile_dim, stride_dim):
        positions = []
        pos = 0
        while pos + tile_dim <= img_dim:
            positions.append(pos)
            pos += stride_dim
        # Add the last position to ensure the end of the image is covered
        # This handles cases where stride doesn't perfectly align with (img_dim - tile_dim)
        if (img_dim - tile_dim) not in positions: # Check if the last possible tile start is not already included
             positions.append(img_dim - tile_dim)
        return sorted(list(set(positions))) # Sort and remove duplicates in case (img_dim - tile_dim) was already added

    y_positions = get_positions(H, tile_h, stride_h)
    x_positions = get_positions(W, tile_w, stride_w)

    for y in y_positions:
        for x in x_positions:
            y1 = y
            y2 = min(y + tile_h, H)
            x1 = x
            x2 = min(x + tile_w, W)

            patch = image_tensor[:, :, y1:y2, x1:x2]
            
            # Ensure patch dimensions are correct for interpolation target
            current_patch_h = y2 - y1
            current_patch_w = x2 - x1

            logits = model(patch.to(device)).logits
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (C, model_output_h, model_output_w)

            # Upsample probs back to the original patch size (y2-y1, x2-x1)
            # Use 'bilinear' for continuous data like probabilities, align_corners=False for consistency
            probs_upsampled = F.interpolate(
                probs.unsqueeze(0), # Add batch dim for interpolate
                size=(current_patch_h, current_patch_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0) # Remove batch dim

            output_probs[:, y1:y2, x1:x2] += probs_upsampled
            count_map[y1:y2, x1:x2] += 1

    # Avoid division by zero
    count_map[count_map == 0] = 1 # This line is good
    output_probs = output_probs / count_map.unsqueeze(0)
    pred_mask = output_probs.argmax(dim=0).cpu()
    return pred_mask

def plot_and_save_contours(im, output_dir, model_name, checkpoint, filename, cam_sys, scalar, n_tiles, coverage, full_pred=None):
    # im is the original image (HWC, BGR, uint8)
    im_f = im.copy()
    im_f = cv2.resize(im_f, (Utils.get_native_dimensions(cam_sys)), interpolation=cv2.INTER_LINEAR)
    if full_pred is not None:
        # full_pred is the prediction mask (H, W) with class indices
        print()
        mask = full_pred.astype(np.uint8)
        mask = cv2.resize(mask, (Utils.get_native_dimensions(cam_sys)), interpolation=cv2.INTER_LINEAR)
        print(mask.shape, "mask shape")
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        im_f = cv2.drawContours(im_f, contours, -1, (255, 0, 255), 5)
    # code for overlay_texts and saving...
    overlay_texts = [
        (f"Coverage: {coverage:.2f}", (10, 30)),
        (f"Tiles: {n_tiles}", (10, 60)),
        (f"Scalar: {scalar:.1f}", (10, 90)),
        (f"Model: {model_name}", (10, 120)),
        (f"Checkpoint: {checkpoint}", (10, 150)),
        (f"Camera: {cam_sys}", (10, 180)),
    ]
    for text, pos in overlay_texts:
        im_f = cv2.putText(im_f, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    output_dir = os.path.join(output_dir, "overlays")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    im_out = os.path.join(output_dir, f"{model_name}_{checkpoint}_{filename}_{cam_sys}_{scalar:0.1f}_{n_tiles}.jpg")
    cv2.imwrite(im_out, im_f)

def save_binary_mask(pred_mask, filename, output_dir):
    # save pred mask as a binary mask png
    binary_mask = pred_mask.squeeze(0).cpu().numpy().astype(np.uint8)
    # binary_mask = cv2.resize(binary_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    binary_mask_folder = os.path.join(output_dir, "binary_masks")
    Utils.setup_directory(binary_mask_folder)
    binary_mask_filename = os.path.join(binary_mask_folder, f"{filename}_mussel_binary_mask.png")
    cv2.imwrite(binary_mask_filename, binary_mask)

def save_results_to_csv(out_csv, filename, coverage, height, width, scalar, n_tiles):
    with open(out_csv, "a", newline='') as f:
        csv.writer(f).writerow([filename, coverage, height, width, scalar, n_tiles])

def main():
    args = parse_arguments()
    formatted_date = datetime.now().strftime('%Y%m%d_%H%M')
    ckpt = args.ckpt_dir
    dirname = os.path.basename(args.image_dir)
    output_dir = os.path.join(args.output_dir, args.output_name, dirname+"_"+str(args.scalar)+"_"+formatted_date)
    scalar = args.scalar
    skip = args.plot_every
    Utils.setup_directory(output_dir)

    model_name = os.path.basename(os.path.dirname(ckpt))
    checkpoint = os.path.basename(ckpt)
    csv_file = f"{model_name}_{checkpoint}_{args.cam_sys}.csv"
    out_csv = os.path.join(output_dir, csv_file)
    imgs_inferred = create_csv_file(out_csv)

    native_width, native_height  = Utils.get_native_dimensions(args.cam_sys)
    height, width = int(native_height * args.scalar), int(native_width * args.scalar)
    resize_dict = {"height": height, "width": width}

    # Calculate tile_size and stride based on tile_prop
    tile_prop = args.tile_prop
    tile_h = int(height * tile_prop)
    tile_w = int(width * tile_prop)
    # Ensure tile size does not exceed image size
    tile_h = min(tile_h, height)
    tile_w = min(tile_w, width)
    tile_size = (tile_h, tile_w)
    # Calculate stride as half of tile size
    stride = (tile_h // 2, tile_w // 2)

    feature_extractor = SegformerImageProcessor(
        do_resize=True,
        size=resize_dict,
        resample=PIL.Image.Resampling.LANCZOS,
        do_reduce_labels=False
    )
    feature_extractor_disp = SegformerImageProcessor(
        do_resize=True,
        size=resize_dict,
        resample=PIL.Image.Resampling.LANCZOS,
        do_reduce_labels=False,
        do_rescale=False,
        do_normalize=False
    )

    # --- SPEEDUP: Use float16 for inference ---
    # Add pin_memory=True for faster host-to-device transfer if using CUDA
    dataset = SemanticSegmentationDatasetDisplay(
        root_dir=args.image_dir,
        feature_extractor=feature_extractor,
        feature_extractor_disp=feature_extractor_disp
    )
    # Use batch_size from args
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(ckpt, device)
    # Convert model to half precision if using CUDA
    if device.type == "cuda":
        model = model.half()

    # If resume, build a set for fast lookup
    imgs_inferred_set = set(imgs_inferred) if args.resume else None

    # Calculate n_tiles using the same logic as sliding_window_inference
    def compute_n_tiles(img_h, img_w, tile_h, tile_w, stride_h, stride_w):
        y_positions = list(range(0, img_h - tile_h + 1, stride_h))
        if y_positions[-1] != img_h - tile_h:
            y_positions.append(img_h - tile_h)
        x_positions = list(range(0, img_w - tile_w + 1, stride_w))
        if x_positions[-1] != img_w - tile_w:
            x_positions.append(img_w - tile_w)
        return len(y_positions) * len(x_positions)

    with torch.no_grad():
        for i, (batch, batch_disp) in tqdm(enumerate(dataloader)):
            filename = dataset.images[i].split(".")[0]
            # skip if already inferred
            if imgs_inferred_set is not None and filename in imgs_inferred_set:
                continue
            # Convert to device and half precision if using CUDA
            if device.type == "cuda":
                pixel_values = batch["pixel_values"].half()

            # Use sliding window inference with user-specified tile_size and stride
            if tile_prop < 1:
                pred_mask = sliding_window_inference(
                    pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values,
                    model,
                    device,
                    tile_size,
                    stride
                )
                n_tiles = compute_n_tiles(height, width, tile_size[0], tile_size[1], stride[0], stride[1])
            else:
                pixel_values = pixel_values.unsqueeze(0) if pixel_values.ndim == 3 else pixel_values
                logits = model(pixel_values.to(device)).logits
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred_mask = probs.argmax(dim=0).cpu()
                n_tiles = 1

            # Calculate coverage
            coverage = pred_mask.float().mean().item()

            if args.plot and i % skip == 0:
                pv = batch_disp["pixel_values"].numpy().squeeze()
                # If im is CHW, convert to HWC for overlay
                if pv.shape[0] in [1, 3]:
                    pv = np.moveaxis(pv, 0, -1)
                if pv.shape[-1] == 3 and pv.dtype != np.uint8:
                    pv = (pv * 255).clip(0, 255).astype(np.uint8)
                # Pass the full prediction mask for overlay
                plot_and_save_contours(
                    pv, output_dir, model_name, checkpoint,
                    filename, args.cam_sys, args.scalar, n_tiles, coverage,
                    full_pred=pred_mask.numpy()
                )
                save_binary_mask(pred_mask, filename, output_dir)

            # Print tiling info for each image
            if i == 0:
                print("performing tiled inference", n_tiles, "tiles")
                print("Tile Size (height, width):", tile_size)
                print("Stride (height, width):", stride)

            save_results_to_csv(out_csv, filename, coverage, height, width, scalar, n_tiles)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    # Print parsed arguments for direct execution
    args = parse_arguments()
    print("Parsed arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
