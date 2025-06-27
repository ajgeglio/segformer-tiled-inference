'''
# This code is part of the MusselFinder project, which aims to provide a robust
# solution for mussel detection and segmentation in underwater images.
#
# The code is designed to calculate the scalar factor for image processing
# The scalar is simply the ratio of the pixel size in the image to the desired pixel size (based on training data).
#
# This class handles loading metadata, substrate data, and images,
# merging dataframes, and calculating the scalar factor based on image properties.
# It also provides methods for cleaning results and calculating pixel sizes.
#
# Copyright (C) 2023-2024 Angus Galloway (agalloway@engtech.ca) 
# Engineering Technologies Canada Ltd.

# Copyright (C) 2023-2025 Anthony Geglio (ajgeglio@mtu.edu)
# Michigan Technological University


'''

import pandas as pd
import numpy as np
import math
import glob 
import os
from utils import return_time
from utils import Utils

class ScalarBase:
    def __init__(self, **kwargs):
        # Accept any keyword arguments for flexibility
        for k, v in kwargs.items():
            setattr(self, k, v)

    def load_OP(self, filepath, **kwargs):
        site_ids = pd.read_excel(filepath).loc[:, ["COLLECT_ID", "SURVEY123_NAME", "LAKE"]]
        site_ids = site_ids.rename(columns={"COLLECT_ID": "CollectID"})
        return site_ids

    def load_metadata(self, filepath, low_memory=False, usable=True, **kwargs):
        meta = pd.read_csv(filepath, low_memory=low_memory)
        meta = meta.dropna(subset=["Latitude", "Longitude", "DistanceToBottom_m"])
        if usable:
            meta = meta[meta["Usability"] == "Usable"]
        return meta.rename(columns={"collect_id": "CollectID", "filename": "Filename", "time_s": "Time_s"})

    def load_substrate(self, filepath, low_memory=False, classes="2c", **kwargs):
        if filepath.endswith('.csv'):
            substrate = pd.read_csv(filepath, index_col=0, low_memory=low_memory)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            substrate = pd.read_pickle(filepath)
        if classes == "2c":
            substrate = substrate[["Filename", "substrate_class_2c"]]
        elif classes == "6c":
            substrate = substrate[["Filename", "substrate_class_6c"]]
        return substrate.drop_duplicates(subset="Filename")

    def merge_dataframes(self, metadata, substrate, site_ids, **kwargs):
        df_combined = pd.merge(metadata, substrate, how='left', on='Filename').sort_values(by="Time_s").reset_index(drop=True)
        df_combined = pd.merge(df_combined, site_ids, on="CollectID", how="left")
        return df_combined

    def clean_results(self, df_combined, **kwargs):
        columns_ = [
            "Filename", "SURVEY123_NAME", "LAKE", "imw", "imh", "Time_s", "Latitude", "Longitude", "DepthFromSurface_m", 
            "DistanceToBottom_m", "Speed_kn", "Time_UTC", "image_path", "CollectID", "REEF", "Year", "Month", "Day", 
            "Time", "ImageArea_m2", "PS_mm", "desired_PS_mm", "SCALAR", 
            "substrate_class_2c"
        ]
        res_df = pd.DataFrame(columns=columns_)
        df = df_combined.loc[:, df_combined.columns.isin(columns_)]
        join_columns = res_df.columns[res_df.columns.isin(df.columns)]
        res_df[join_columns] = df[join_columns]
        res_df["Year"] = res_df["Time_s"].apply(return_time().get_Y)
        res_df["Month"] = res_df["Time_s"].apply(return_time().get_m)
        res_df["Day"] = res_df["Time_s"].apply(return_time().get_d)
        res_df["Time"] = res_df["Time_s"].apply(return_time().get_t)
        res_df = res_df.reset_index(drop=True)
        return res_df

    def angular_field_of_view(self, h, f, **kwargs):
        # Returns angular field of view in degrees
        return math.degrees(math.atan(h / f))

    def horizontal_field_of_view(self, AFOV, W, **kwargs):
        # Returns horizontal field of view in mm
        return 2 * W * math.tan(math.radians(AFOV / 2))

    def size_to_pixel(self, HFOV, N, **kwargs):
        return HFOV / N

    def calculate_scalar(self, PS_mm, desired_PS_mm, **kwargs):
        # Calculate the scalar based on the pixel size and desired pixel size
        return PS_mm / desired_PS_mm

    def calculate_indices(self, res_df, **kwargs):
        collect_id = res_df['CollectID']
        ABS = collect_id.apply(lambda x: x.split('_')[-1])
        CAM = collect_id.apply(lambda x: x.split('_')[-2])
        CD = collect_id.apply(lambda x: int(x.split('_')[0]))

        indices = {
            "ABS1_idx": collect_id[ABS == "ABS1"].index,
            "ABS2_idx": collect_id[ABS == "ABS2"].index,
            "Iver3069_idx": collect_id[CAM == "Iver3069"].index,
            "Iver3098_idx": collect_id[CAM == "Iver3098"].index,
            "ABS2_3069_12mm_idx": collect_id[ABS == "ABS2"].index.intersection(collect_id[CAM == "Iver3069"].index).intersection(CD[CD >= 20220716].index).intersection(CD[CD < 20220822].index),
            "ABS2_3069_16mm_idx": collect_id[ABS == "ABS2"].index.intersection(collect_id[CAM == "Iver3069"].index).intersection(CD[CD >= 20220822].index),
            "ABS2_3098_12mm_idx": collect_id[ABS == "ABS2"].index.intersection(collect_id[CAM == "Iver3098"].index).intersection(CD[CD>=20220706].index).intersection(CD[CD< 20220819].index),
            "ABS2_3098_16mm_idx": collect_id[ABS == "ABS2"].index.intersection(collect_id[CAM == "Iver3098"].index).intersection(CD[CD>=20220819].index)
        }
        return indices

    def area_and_pixel_size(self, res_df, **kwargs):
        res_df = res_df.copy()
        l = len(res_df)
        # Initialize arrays
        h_ = np.full(l, np.nan)
        f = np.full(l, np.nan)
        imgh = np.full(l, np.nan)
        imgw = np.full(l, np.nan)
        W = res_df['DistanceToBottom_m'].values * 1000  # Convert to mm

        N = 4112  # Number of horizontal pixels

        indices = self.calculate_indices(res_df)

        # Set image dimensions
        imgh[indices["ABS1_idx"]] = 2176
        imgw[indices["ABS1_idx"]] = 4096
        imgh[indices["ABS2_idx"]] = 3000
        imgw[indices["ABS2_idx"]] = 4096

        # Set sensor heights
        h_[indices["ABS1_idx"]] = 14.2
        h_[indices["ABS2_idx"]] = 14.1

        # Set focal lengths
        f[indices["ABS1_idx"]] = 16
        f[indices["ABS2_3069_12mm_idx"]] = 12
        f[indices["ABS2_3069_16mm_idx"]] = 16
        f[indices["ABS2_3098_12mm_idx"]] = 12
        f[indices["ABS2_3098_16mm_idx"]] = 16

        # Calculate fields of view and pixel size
        AFOV = np.degrees(np.arctan(h_ / f))
        HFOV = 2 * W * np.tan(np.radians(AFOV / 2))
        PS = HFOV / N

        # Calculate image area
        image_ratio = imgh / imgw
        VFOV = image_ratio * HFOV
        area = (HFOV * VFOV) / 1e6  # m^2

        # Assign results
        res_df["imh"] = imgh.astype(int)
        res_df["imw"] = imgw.astype(int)
        res_df["PS_mm"] = PS
        res_df["ImageArea_m2"] = area

        return res_df

    def output_dimensions(self, res_df, desired_PS_mm=None, **kwargs):
        res_df = res_df.copy()
        desired_PS_mm = desired_PS_mm if desired_PS_mm is not None else self.desired_PS_mm
        PS_mm = res_df.PS_mm
        SCALAR_func = np.vectorize(self.calculate_scalar)
        SCALAR = SCALAR_func(PS_mm, desired_PS_mm)
        res_df.SCALAR = SCALAR
        res_df.desired_PS_mm = desired_PS_mm
        return res_df

    def res_final(self, df_combined, desired_PS_mm=None, **kwargs):
        print("Cleaning Results Spreadsheet", end="\r")
        res_df = self.clean_results(df_combined)
        print("calculating image are and pixel size", end="\r")
        res_df = self.area_and_pixel_size(res_df)
        res_df = self.output_dimensions(
            res_df,
            desired_PS_mm=desired_PS_mm
        )
        return res_df

class CalculateScalar(ScalarBase):
    def __init__(
        self,
        img_folder_pths=None,
        meta_csv_pths=None,
        substrate_path=None,
        OP_table_path=None,
        desired_PS_mm=0.425,
        **kwargs
    ):
        super().__init__(
            img_folder_pths=img_folder_pths,
            meta_csv_pths=meta_csv_pths,
            substrate_path=substrate_path,
            OP_table_path=OP_table_path,
            desired_PS_mm=desired_PS_mm,
            **kwargs
        )
        self.img_folder_pths = img_folder_pths
        self.meta_csv_pths = meta_csv_pths
        self.substrate_path = substrate_path
        self.OP_table_path = OP_table_path
        self.desired_PS_mm = desired_PS_mm

    def initialize_metdatadata(self, meta_csv_pths=None, img_folder_pths=None, **kwargs):
        meta_csv_pths = meta_csv_pths if meta_csv_pths is not None else self.meta_csv_pths
        img_pths = img_folder_pths if img_folder_pths is not None else self.img_folder_pths
        meta_df = pd.concat([self.load_metadata(pth) for pth in meta_csv_pths], ignore_index=True)
        glob_pths = lambda pth: pd.DataFrame(glob.glob(os.path.join(pth, "*.png")), columns=["image_path"])
        image_pth_df = pd.concat([glob_pths(pth) for pth in img_pths], ignore_index=True)
        image_pth_df["Filename"] = image_pth_df["image_path"].apply(lambda x: os.path.basename(x).split(".")[0])
        meta_df = pd.merge(meta_df, image_pth_df, on="Filename", how="left")
        meta_df = meta_df.rename(columns={"collect_id": "CollectID", "filename": "Filename", "time_s": "Time_s"})
        return meta_df

    def combine_meta_pred_substrate(self, **kwargs):
        site_ids = self.load_OP(self.OP_table_path)
        substrate = self.load_substrate(self.substrate_path)
        metadata = self.initialize_metdatadata(meta_csv_pths=self.meta_csv_pths, img_folder_pths=self.img_folder_pths)
        print("substrate images pred", substrate.shape)
        print("metadata csv", metadata.shape)
        df_combined = self.merge_dataframes(metadata, substrate, site_ids)
        print("final", df_combined.shape)
        assert metadata.shape[0] == df_combined.shape[0]
        return df_combined

    def scalar_inference_metadata(self, desired_PS_mm=None, **kwargs):
        df_combined = self.combine_meta_pred_substrate()
        # Use defaults if not provided
        if desired_PS_mm is None:
            desired_PS_mm = self.desired_PS_mm if hasattr(self, "desired_PS_mm") else 0.425
        scalar_inference_df = self.res_final(
            df_combined,
            desired_PS_mm=desired_PS_mm
        )
        assert scalar_inference_df.shape == scalar_inference_df.dropna(subset=["Latitude", "Longitude", "DistanceToBottom_m"]).shape
        return scalar_inference_df

if __name__ == "__main__":
    # Example file paths (replace with your actual file paths)
    meta_csv_pths = ["path/to/metadata1.csv", "path/to/metadata2.csv"]
    # List of image folder paths
    img_folder_pths = ["path/to/image_folder1", "path/to/image_folder2"]
    # Substrate and OP table paths
    substrate_path = "path/to/substrate.csv"
    OP_table_path = "path/to/OP_table.xlsx"

    # Create an instance of CalculateScalar with all paths as arguments
    calc = CalculateScalar(
        img_folder_pths=img_folder_pths,
        meta_csv_pths=meta_csv_pths,
        substrate_path=substrate_path,
        OP_table_path=OP_table_path
    )

    musselfinder_inference = calc.musselfinder_inference_metadata(
        desired_PS_mm=0.425
    )

    # Print or save the result
    print(musselfinder_inference.head())
