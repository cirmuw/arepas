#!/usr/bin/env python3
import numpy as np
import torch
import utils
from config import PATCH_SIZE


def generate_final_anomaly_heatmap(ind, X_test, Y_test, generator, siamese):
    """
    Generate an anomaly heatmap for a given test image index using PyTorch models.

    Parameters
    ----------
    ind : int
        Index of the test image to process.
    X_test, Y_test : np.ndarray
        Arrays of shape (N, H, W, C) in [-1, 1].
    generator : torch.nn.Module
        Generator model (expects NCHW input, returns NCHW output).
    siamese : torch.nn.Module
        Siamese model used for patch-level anomaly scoring.
    """

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # --------------------------------------------------
    # Prepare single test sample
    # --------------------------------------------------
    inp = np.array(X_test[ind:ind+1])  # (1, H, W, C)
    real = np.array(Y_test[ind:ind+1])  # (1, H, W, C)

    # NHWC -> NCHW for PyTorch
    inp_t = torch.from_numpy(np.transpose(inp, (0, 3, 1, 2))).float().to(device)

    # --------------------------------------------------
    # Generate prediction with the PyTorch generator
    # --------------------------------------------------
    generator.eval()
    with torch.no_grad():
        pred_t = generator(inp_t)

    # Back to NumPy NHWC for utils functions
    predict = pred_t.detach().cpu().numpy()
    predict = np.transpose(predict, (0, 2, 3, 1))  # (1, H, W, C)

    # --------------------------------------------------
    # Compute anomaly heatmap via utils (same as TF)
    # --------------------------------------------------
    heat_map = np.zeros((1, 256, 256, 1), dtype=np.float32)

    reassembled_image = utils.process_images_random(
        real,
        predict,
        10000,
        siamese,
        patch_size=(PATCH_SIZE, PATCH_SIZE)
    )

    heat_map += reassembled_image
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map) + 1e-8)

    # Combine heat map with absolute pixel difference
    abs_diff = np.abs(predict[0, :, :, 0] - real[0, :, :, 0])
    final_map = heat_map[0, :, :, 0] * abs_diff

    return final_map
