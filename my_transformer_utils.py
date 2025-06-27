'''
# Utility functions for loading and processing images
# includes dataset classes for semantic segmentation using PyTorch utilities

# Copyright (C) 2023-2024 Angus Galloway (agalloway@engtech.ca) 
# Engineering Technologies Canada Ltd.
#

'''

from transformers.image_utils import ChannelDimension
from torchvision.transforms import ColorJitter
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os
import numpy as np
import torchvision.transforms as T


class SemanticSegmentationTrainValDatasetDA(Dataset):
    """Image (semantic) segmentation dataset, extracts validation set from train set"""

    def __init__(
        self,
        root_dir,
        feature_extractor,
        split="train",
        val_pct=20,
        seed=1234,
        image_suffix=None,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            split: train or val
            val_pct; percentage of training set to reserve for validation
            seed: numpy random seed for shuffling train/val set, must be the same for train and val
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.split = split
        self.seed = seed
        self.image_suffix = image_suffix

        np.random.seed(seed)

        if self.split == "train":
            self.jitter = ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
            )
            self.hflip = T.RandomHorizontalFlip(
                0.5
            )  # rotate image about y-axis with 50% prob
            self.vflip = T.RandomVerticalFlip(0.5)
            self.rot = T.RandomRotation([0, 90, 180, 270, 360])

        self.img_dir = os.path.join(self.root_dir, "images")
        print(self.img_dir)
        self.img_dir = os.path.join(self.root_dir, "images" + image_suffix)
        print(self.img_dir)
        self.ann_dir = os.path.join(self.root_dir, "annotations")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

        # print(self.annotations[:3])

        # draw random train/val split
        n = len(self.images)
        indices = np.arange(n)
        np.random.shuffle(indices)
        # print(indices)

        # convert pct % to index
        cutoff = int(n * val_pct / 100)
        # print(cutoff)

        if self.split == "train":
            # take the first 100 - val_pct as train set
            self.annotations = list(np.asarray(self.annotations)[indices][:-cutoff])
            self.images = list(np.asarray(self.images)[indices][:-cutoff])
        else:
            # take the last val_pct as val set
            self.annotations = list(np.asarray(self.annotations)[indices][n - cutoff :])
            self.images = list(np.asarray(self.images)[indices][n - cutoff :])

        # print(self.annotations[:3])
        # print(len(self.annotations))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # data aug
        if self.split == "train":
            image = self.jitter(image)
            image, segmentation_map = self.hflip(image, segmentation_map)
            image, segmentation_map = self.vflip(image, segmentation_map)
            image, segmentation_map = self.rot(image, segmentation_map)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SemanticSegmentationDatasetDAFromList(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self, data_dirs, feature_extractor, split=None, limit=0, do_data_aug=False
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.data_dirs = data_dirs
        self.feature_extractor = feature_extractor
        self.split = split
        self.do_data_aug = do_data_aug
        self.images = []
        self.annotations = []

        if self.do_data_aug:
            self.jitter = ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
            )
            self.hflip = T.RandomHorizontalFlip(
                0.5
            )  # rotate image about y-axis with 50% prob
            self.vflip = T.RandomVerticalFlip(0.5)
            self.rot = T.RandomRotation([0, 90, 180, 270])

        # loop over all datasets
        for data_dir in data_dirs:
            img_dir = os.path.join(data_dir, "images")
            ann_dir = os.path.join(data_dir, "annotations")

            # read images
            images = sorted(glob(img_dir + "/*.jpg"))
            self.images += images

            # read annotations
            annotations = sorted(glob(ann_dir + "/*.png"))
            self.annotations += annotations

        if limit:
            self.images = self.images[:limit]
            self.annotations = self.annotations[:limit]

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # this part is problematic, need to know which img_dir to use...
        image = Image.open(os.path.join(self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.annotations[idx]))

        # data aug
        if self.do_data_aug:
            image = self.jitter(image)
            image, segmentation_map = self.hflip(image, segmentation_map)
            image, segmentation_map = self.vflip(image, segmentation_map)
            image, segmentation_map = self.rot(image, segmentation_map)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SemanticSegmentationDatasetDA(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self,
        root_dir,
        feature_extractor,
        split=None,
        seek=0,
        limit=0,
        do_data_aug=False,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            split: optional partition subfolder under root_dir (e.g., train, val, test),
                   this is generally not used anymore so that the splits can be determined at run time.
            seek: start index for sampling, discards all samples with index < seek.
            limit: stop index for sampling, discards all samples with index > limit.
            do_data_aug: whether to apply data augmentation (usually True for train set, valse for val set).
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.split = split
        self.do_data_aug = do_data_aug

        if self.do_data_aug:
            self.jitter = ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
            )
            self.hflip = T.RandomHorizontalFlip(
                0.5
            )  # rotate image about y-axis with 50% prob
            self.vflip = T.RandomVerticalFlip(0.5)
            self.rot = T.RandomRotation([0, 90, 180, 270, 360])

        # https://unix.stackexchange.com/questions/29214/copy-first-n-files-in-a-different-directory

        #
        # self.img_dir = os.path.join(self.root_dir, "JPEGImages")
        # self.ann_dir = os.path.join(self.root_dir, "SegmentationClass")

        if self.split is not None:
            self.img_dir = os.path.join(self.root_dir, self.split, "images")
            self.ann_dir = os.path.join(self.root_dir, self.split, "annotations")
        else:
            self.img_dir = os.path.join(self.root_dir, "images")
            self.ann_dir = os.path.join(self.root_dir, "annotations")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        if seek:
            self.images = self.images[seek:]
            self.annotations = self.annotations[seek:]

        if limit:
            self.images = self.images[:limit]
            self.annotations = self.annotations[:limit]

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # data aug
        if self.do_data_aug:
            image = self.jitter(image)
            image, segmentation_map = self.hflip(image, segmentation_map)
            image, segmentation_map = self.vflip(image, segmentation_map)
            image, segmentation_map = self.rot(image, segmentation_map)

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(
        self,
        root_dir,
        feature_extractor,
        split=None,
        limit=0,
        annotation_set=None,
        image_suffix=None,
    ):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.split = split
        self.annotation_set = annotation_set
        self.image_suffix = image_suffix

        # https://unix.stackexchange.com/questions/29214/copy-first-n-files-in-a-different-directory

        #
        # self.img_dir = os.path.join(self.root_dir, "JPEGImages")
        # self.ann_dir = os.path.join(self.root_dir, "SegmentationClass")

        if self.split is not None:
            self.img_dir = os.path.join(
                self.root_dir, self.split, "images" + self.image_suffix
            )
            self.ann_dir = os.path.join(self.root_dir, self.split, "annotations")
        else:
            self.img_dir = os.path.join(self.root_dir, "images" + self.image_suffix)
            if self.annotation_set is not None:
                self.ann_dir = os.path.join(
                    self.root_dir, str(self.annotation_set) + "_annotations"
                )
            else:
                self.ann_dir = os.path.join(self.root_dir, "annotations")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        if limit:
            # self.images = self.images[:limit]
            self.annotations = self.annotations[:limit]
            # annotations may be in random order, safest thing to do is
            # generate the list of images from the annotations
            self.images = [sub.replace("png", "jpg") for sub in self.annotations]

        assert len(self.images) == len(
            self.annotations
        ), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(
            image, segmentation_map, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SemanticSegmentationDatasetNoLabels(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, split=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.split = split

        if split is not None:
            self.img_dir = os.path.join(self.root_dir, self.split)
        else:
            self.img_dir = self.root_dir

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        # segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        # encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt", do_resize=False)
        encoded_inputs = self.feature_extractor(
            image, return_tensors="pt", do_resize=False
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SemanticSegmentationDatasetDisplay(Dataset):
    """Image (semantic) segmentation dataset, returns normalized
    and unnormalized copy of the same image for feeding to DNN and drawing contours.

    This dataloader was initially created for the MusselFinder_ABS2 notebook"""

    def __init__(self, root_dir, feature_extractor, feature_extractor_disp, split=None):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images for SegFormer
            feature_extractor_disp (SegFormerFeatureExtractor): feature extractor to prepare images for adding contours
            split: whether to load jpeg or png
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.feature_extractor_disp = feature_extractor_disp
        self.split = split

        if split is not None:
            self.img_dir = os.path.join(self.root_dir, self.split)
        else:
            self.img_dir = self.root_dir

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))

        encoded_inputs = self.feature_extractor(image, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        encoded_inputs_disp = self.feature_extractor_disp(
            image, return_tensors="np", data_format=ChannelDimension.LAST
        )

        for k, v in encoded_inputs_disp.items():
            encoded_inputs_disp[k].squeeze()  # remove batch dimension

        return encoded_inputs, encoded_inputs_disp
