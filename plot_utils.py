"""Plotting utilities for MusselFinder"""
#
# Copyright (C) 2023-2024 Angus Galloway (agalloway@engtech.ca) 
# Engineering Technologies Canada Ltd.


from skimage import measure
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import cv2


def draw_pink_contours(input_image, predictions):
    contours, _ = cv2.findContours(
        predictions.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    return cv2.drawContours(
        input_image,
        contours,
        -1,
        (255, 0, 255),
        5,
    )


def get_unique_filename(o, m, e):
    """
    o: output directory
    m: model checkpoint name
    e: data directory
    """
    return "prediction_" + o.split("/")[-1] + "_" + m + "_" + e.split("/")[-1]


def get_unique_filename_no_metrics(o, m, e, img_file, img_size, suffix=None):
    """
    o: output directory
    m: model checkpoint name
    e: data directory
    """
    f = (
        "prediction_"
        + o.split("/")[-1]
        + "_"
        + m
        + "_"
        + e.split("/")[-1]
        + "_"
        + img_file
        + "_%dpx" % img_size
    )
    if suffix is not None:
        f += "_" + suffix
    return f


def plot_1x3_predictions(img, p, l, alpha=0.3):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    p3chan = np.concatenate(
        (
            np.zeros((p.shape[0], p.shape[1], 1)),
            np.zeros((p.shape[0], p.shape[1], 1)),
            np.expand_dims(p, 2),
        ),
        axis=2,
    )

    # process dot labels for viz
    l[l == 1] = 2
    l[l == 255] = 1

    viz = img.transpose(1, 2, 0)
    viz = viz - viz.min()
    viz = viz / viz.max()

    overlay = cv2.addWeighted(p3chan.astype("float32"), alpha, viz, 1 - alpha, 0)

    ax[0].imshow(viz)
    ax[0].set_title("Input " + str(i))
    ax[1].imshow(overlay)
    ax[1].set_title("Prediction")
    ax[2].imshow(l)
    ax[2].set_title("Label")
    for j in range(3):
        ax[j].axis("off")
    plt.show()

    filename = get_unique_filename(output_dir, detailed_model, eval_data_dir)

    # fig.savefig(osp.join('img/b3-2000', filename + '_%d' % i), bbox_inches='tight')
    # print('Saved ', filename + '_%d' % i)


def plot_1x1_predictions(save_dir, img, p, metrics, f, i, alpha=0.3, save=False):
    """
    Works with SegFormer_Inference notebook
    """

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    p3chan = np.concatenate(
        (
            np.zeros((p.shape[0], p.shape[1], 1)),
            np.zeros((p.shape[0], p.shape[1], 1)),
            np.expand_dims(p, 2),
        ),
        axis=2,
    )

    viz = img.transpose(1, 2, 0)
    viz = viz - viz.min()
    viz = viz / viz.max()

    overlay = cv2.addWeighted(p3chan.astype("float32"), alpha, viz, 1 - alpha, 0)

    ax.imshow(overlay)

    title = "IMG %d: mean IoU %.1f%%, mussel IoU %.1f%%, bkg IoU %.1f%%" % (
        i,
        metrics["mean_iou"] * 100,
        metrics["iou_mussel"] * 100,
        metrics["iou_background"] * 100,
    )

    ax.set_title(title)
    ax.axis("off")
    plt.show()

    if save:
        fig.savefig(
            osp.join(save_dir, f + "_%d_%.4f.jpg" % (i, metrics["mean_iou"])),
            bbox_inches="tight",
            format="jpg",
        )
        print("Saved ", f + "_%d" % i)


def plot_1x1_error_metrics(save_dir, lab, p, metrics, f, i, save=False):
    """
    Works with SegFormer_Inference notebook
    """

    iou_metrics = lab + p * 2

    error_types = ["TN", "FN", "FP", "TP"]

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    im = ax.imshow(
        iou_metrics.squeeze(), cmap=plt.cm.get_cmap("inferno", len(error_types))
    )

    title = "IMG %d: mean IoU %.1f%%, mussel IoU %.1f%%, bkg IoU %.1f%%" % (
        i,
        metrics["mean_iou"] * 100,
        metrics["iou_mussel"] * 100,
        metrics["iou_background"] * 100,
    )
    ax.set_title(title)

    # This function formatter will replace integers with target names
    formatter = plt.FuncFormatter(lambda x, loc: error_types[x])

    cbar = fig.colorbar(
        im,
        ax=ax,
        ticks=[0, 1, 2, 3],
        format=formatter,
        orientation="horizontal",
        shrink=0.4,
        pad=0.05,
        aspect=30,
        fraction=0.05,
    )
    cbar.ax.tick_params(labelsize=18)

    plt.show()

    if save:
        fig.savefig(
            osp.join(save_dir, f + "_%d_%.4f_metrics.jpg" % (i, metrics["mean_iou"])),
            bbox_inches="tight",
            format="jpg",
        )
        print("Saved ", f + "_%d" % i)


def plot_1x1_predictions_no_metrics(
    save_dir, img, p, f, i, alpha=0.3, do_contour=True, save=False
):
    """
    param save_dir: (string) path to save image if `save=True`
    param img: (nparray) raw color image (3, H, W)
    param p: (nparray) prediction matrix (1, H, W)
    param f: (string) filename
    param i: (int) index counter
    param alpha: (float) weighting for overlay
    param do_contour: (boolean) whether to display prediction as contours or overlay
    param save: (boolean) whether to save the image to disk or not
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    viz = img.transpose(1, 2, 0)
    viz = viz - viz.min()
    viz = viz / viz.max()

    if do_contour:
        ax.imshow(viz)
        contours = measure.find_contours(p, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, c="lime")
    else:
        p3chan = np.concatenate(
            (
                np.zeros((p.shape[0], p.shape[1], 1)),
                np.zeros((p.shape[0], p.shape[1], 1)),
                np.expand_dims(p, 2),
            ),
            axis=2,
        )
        overlay = cv2.addWeighted(p3chan.astype("float32"), alpha, viz, 1 - alpha, 0)
        ax.imshow(overlay)

    title = "IMG %d: Coverage %.1f%%" % (i, 100 * p.mean())
    ax.set_title(title)
    ax.axis("off")
    # plt.show()

    if save:
        fig.savefig(
            osp.join(save_dir, f + "_%d.jpg" % i), bbox_inches="tight", format="jpg"
        )
        print("Saved ", f + "_%d" % i)
