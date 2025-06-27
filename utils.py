"""
# This file contains utility functions and classes for MusselFinder

# image segmentation and processing. It includes functions for copying scalar bins,
# converting images, and creating unique filenames. It also defines a custom dataset class
# for image segmentation using a feature extractor. The code is designed to work with
# PyTorch and OpenCV for image processing and manipulation.

# Copyright (C) 2023-2024 Angus Galloway (agalloway@engtech.ca) 
# Engineering Technologies Canada Ltd.

# Copyright (C) 2023-2025 Anthony Geglio (ajgeglio@mtu.edu)
# Michigan Technological University

"""
from torch.utils.data import Dataset
from PIL import Image
import glob
import cv2
import datetime
import shutil
import os
import pandas as pd
import csv

class return_time:
    def __init__(self) -> None:
        pass
    def get_time_obj(self, time_s):
        return datetime.datetime.fromtimestamp(time_s) if pd.notnull(time_s) else np.nan
    def get_Y(self, time_s):
        return self.get_time_obj(time_s).strftime('%Y') if pd.notnull(time_s) else np.nan
    def get_m(self, time_s):
        return self.get_time_obj(time_s).strftime('%m') if pd.notnull(time_s) else np.nan
    def get_d(self, time_s):
        return self.get_time_obj(time_s).strftime('%d') if pd.notnull(time_s) else np.nan
    def get_t(self, time_s):
        return self.get_time_obj(time_s).strftime('%H:%M:%S') if pd.notnull(time_s) else np.nan
    
class Utils:
    def __init__(self):
        pass

    def setup_directory(output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_native_dimensions(cam_sys):
        dimensions = {
            "Abiss1": (4096, 2176),
            "Abiss2": (4096, 3000),
            "REMUS": None
        }
        if cam_sys == "REMUS":
            print("not yet tested for REMUS")
            quit()
        return dimensions[cam_sys]

    def create_inference_csv_file(out_csv):
        try:
            with open(out_csv, "x", newline='') as f:
                csv.writer(f).writerow(["Filename", "Coverage", "infer_imh", "infer_imw", "infer_scalar", "n_tiles"])
                return []
        except FileExistsError:
            with open(out_csv, "r") as f:
                return [row.split(",")[0] for row in f]

    @staticmethod
    def return_YMD():
        """
        Returns a function that outputs the current date in 'YYYYMMDD' format.
        """
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d')

    def copy_scalar_bins(df_SCALAR, cam_sys = "Abiss2", dest = "."):
        bins_ = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 10]
        for a, b in zip(bins_[:-1], bins_[1:]):
            if cam_sys == "Abiss1":
                df = df_SCALAR[df_SCALAR.imh == 2176]
            elif cam_sys == "Abiss2":
                df = df_SCALAR[df_SCALAR.imh == 3000]
            else:
                print("choose camera system")
                break
            df = df[(df.SCALAR>=a)&(df.SCALAR<b)]
            dest_folder = os.path.join(dest, cam_sys, f"Scalar{a}_{b}")
            os.makedirs(dest_folder, exist_ok=True)  # Ensure the destination folder exists
            l = len(df)
            print("copying imgs by SCALARS:", a, "-", b, l, "images to", dest_folder)
            i = 0
            for src in df.image_path:
                f = os.path.basename(src)
                dst = os.path.join(dest_folder, f)
                # src/dest size comparison
                if os.path.exists(src):
                    if os.path.exists(dst):
                        if os.stat(src).st_size == os.stat(dst).st_size:
                            i += 1
                        else:
                            shutil.copy(src, dst)
                            i += 1
                    else:
                        shutil.copy(src, dst)
                        i += 1
                    print("Copying", i, "/", l, end='  \r')
                else:
                    print(f"{src} not found")

    @staticmethod
    def get_file_name(file_path):
        """
        Get the file name without the extension from a given file path.

        """
        return os.path.splitext(os.path.basename(file_path))[0]

    @staticmethod
    def get_file_extension(file_path):
        """
        Get the file extension from a given file path.
   
        """
        return os.path.splitext(file_path)[1][1:]
    
    def convert_imgs(src, dst, fr="png", to="jpg"):
        # Load .png image
        imgs = glob.glob(os.path.join(src,"*"+fr))
        for im in imgs:
            image = cv2.imread(im)
            bn = os.path.basename(im).split(".")[0]
            # Save .jpg (to) image
            cv2.imwrite(os.path.join(dst,f"{bn}.{to}"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def get_unique_filename(m, e, img_file, img_size, suffix=None):
        """
        m: model checkpoint name
        e: save path
        img_file: image filename
        img_size: image width
        suffix: optional suffix
        """
        if not os.path.exists(e):
            os.makedirs(e)
        f = (os.path.join(
            e,
            "prediction_"+img_file+ "_%dpx" % img_size
        ))
        if suffix is not None:
            f += "_" + suffix
        return f
    
    def combine_outputs(self, dest_dir):
        output_csvs = glob.glob(os.path.join(dest_dir, "MusselFinder_Inference_*", "*.csv"))
        combined_df = pd.DataFrame()
        for csv_file in output_csvs:
            df = pd.read_csv(csv_file)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        return combined_df
    
    def combine_inference_metadata(self, dest_dir, metadata_df):
        combined_inference_df = self.combine_outputs(dest_dir)
        metadata_outputs_df = pd.merge(metadata_df, combined_inference_df, on="Filename", how="left")
        metadata_outputs_df = metadata_outputs_df.sort_values(by=["n_tiles"], ascending=True)
        metadata_outputs_df = metadata_outputs_df.drop_duplicates(subset=["Filename"], keep='first')
        metadata_outputs_df = metadata_outputs_df.sort_values(by="Filename")
        metadata_outputs_df = metadata_outputs_df.reset_index(drop=True)
        return metadata_outputs_df
    
class SegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, path, feature_extractor):
        """
        Args:
            path (string): directory of the dataset containing the images.
            feature_extractor (SegFormerImageProcessor): feature extractor to prepare images.
        """
        self.img_dir = path
        self.feature_extractor = feature_extractor

        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))

        encoded_inputs = self.feature_extractor(
            image, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs


