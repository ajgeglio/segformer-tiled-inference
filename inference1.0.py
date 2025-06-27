import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os
import csv
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import PIL
import cv2
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from my_transformer_utils import SemanticSegmentationDatasetDisplay
from plot_utils import draw_pink_contours
# from src.plot_utils import draw_pink_contours
from utils import Utils

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
    parser.add_argument("--divide", help="number to divide height and width of image (2 = 4 tiles, 3 = 9 tiles)", type=int, default=1)
    parser.add_argument("--plot", help="plot contours overlays", action="store_true")
    parser.add_argument("--plot_every", help="plot every x image overlays", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size for inference dataloader", type=int, default=1)
    return parser.parse_args()

def initialize_model(ckpt, device):
    model = SegformerForSemanticSegmentation.from_pretrained(ckpt)
    model.to(device)
    model.eval()
    return model

def process_image_tiles(pixel_values, patch_size, model, device):
    pixel_values = torch.nn.functional.unfold(pixel_values, kernel_size=patch_size, stride=patch_size)
    pixel_values = pixel_values.permute(0, 2, 1).reshape(-1, 3, patch_size[0], patch_size[1])
    outputs = model(pixel_values.to(device))
    return outputs.logits, pixel_values.shape[0]

def save_results_to_csv(out_csv, filename, coverage, height, width, scalar, n_tiles):
    with open(out_csv, "a", newline='') as f:
        csv.writer(f).writerow([filename, coverage, height, width, scalar, n_tiles])

def plot_and_save_contours(im_cont_lst, div, output_dir, model_name, checkpoint, key, cam_sys, scalar, n_tiles, coverage):
    # convert contour list to BGR
    im_cont_lst = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in im_cont_lst]
    # # get the index ranges (from:to) of horizontal rows tiles, i.e. for division by 2...
    # # idx_h = [[0, 2], [2, 4]]
    idx_h = [[i * div + j * div for j in range(2)] for i in range(div)]
    # # perform horizontal concatenation taking the index ranges
    # # and concatenate the images in the list
    h_concats = [cv2.hconcat(im_cont_lst[idx[0]:idx[-1]]) for idx in idx_h]

    # perform vertical concatenation of the horizontal rows
    im_f = cv2.vconcat(h_concats)
    im_f = cv2.resize(im_f, (Utils.get_native_dimensions(cam_sys)), interpolation=cv2.INTER_LINEAR)
    im_f = cv2.cvtColor(im_f, cv2.COLOR_RGB2BGR)
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
    im_out = os.path.join(output_dir, f"{model_name}_{checkpoint}_{key}_{cam_sys}_{scalar:0.1f}_{n_tiles}.jpg")
    cv2.imwrite(im_out, im_f)

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
    imgs_inferred = Utils.create_inference_csv_file(out_csv)

    native_width, native_height = Utils.get_native_dimensions(args.cam_sys)
    height, width = int(native_height * args.scalar), int(native_width * args.scalar)
    resize_dict = {"height": height, "width": width}
    div = args.divide
    patch_size = (height // div, width // div)

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

    with torch.no_grad():
        for i, (batch, batch_disp) in tqdm(enumerate(dataloader)):
            filename = dataset.images[i].split(".")[0]
            if args.resume and filename in imgs_inferred:
                continue

            # Convert input to float16 if using CUDA for speedup
            pixel_values = batch["pixel_values"]
            if device.type == "cuda":
                pixel_values = pixel_values.half()
            logits, n_tiles = process_image_tiles(pixel_values, patch_size, model, device)
            predicted = torch.nn.functional.interpolate(
                logits, size=patch_size, mode="bilinear", align_corners=False
            ).argmax(dim=1)

            tile_covrg_lst = [predicted[i].float().mean().item() for i in range(n_tiles)]
            coverage = sum(tile_covrg_lst) / n_tiles

            if args.plot and i % skip == 0:
                im = batch_disp["pixel_values"].numpy().squeeze()

                # works only for even divisible number of tiles
                # im_lst = [
                #     im[x: x + patch_size[0], y: y + patch_size[1]]
                #     for x in range(0, height, patch_size[0])
                #     for y in range(0, width, patch_size[1])
                # ]

                # Compute tile boundaries to ensure all pixels are included, even if not divisible
                x_edges = [round(i * height / div) for i in range(div + 1)]
                y_edges = [round(i * width / div) for i in range(div + 1)]
                im_lst = [
                    im[x_edges[i]:x_edges[i+1], y_edges[j]:y_edges[j+1]]
                    for i in range(div)
                    for j in range(div)
                ]
                im_cont_lst = [
                    draw_pink_contours(im_lst[i], predicted[i].detach().cpu().numpy().squeeze())
                    for i in range(n_tiles)
                ]
                plot_and_save_contours(
                    im_cont_lst, div, output_dir, model_name, checkpoint,
                    filename, args.cam_sys, args.scalar, n_tiles, coverage
                )

            save_results_to_csv(out_csv, filename, coverage, height, width, scalar, n_tiles)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
