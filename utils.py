#!/usr/bin/env python
# coding: utf-8
import numpy as np


# -----------------------------
# Patch sampling & extraction
# -----------------------------
def _sample_coords(num_patches: int, height: int, width: int, ph: int, pw: int, rng: np.random.Generator):
    """Return (num_patches, 2) array of (y, x) top-left coords."""
    ys = rng.integers(0, height - ph + 1, size=num_patches)
    xs = rng.integers(0, width - pw + 1, size=num_patches)
    return np.stack([ys, xs], axis=1)


def extract_patches_random(images: np.ndarray, patch_size: int, patches_per_image: int, seed: int = 42):
    """
    Randomly extract patches from a batch of images.

    Parameters
    ----------
    images : (N, H, W, C)
    patch_size : int
    patches_per_image : int
    seed : int

    Returns
    -------
    patches : (N * patches_per_image, patch_size, patch_size, C)
    """
    rng = np.random.default_rng(seed)
    n, h, w, c = images.shape
    ph = pw = patch_size

    patches = np.empty((n * patches_per_image, ph, pw, c), dtype=images.dtype)
    k = 0
    for img in images:
        coords = _sample_coords(patches_per_image, h, w, ph, pw, rng)
        for y, x in coords:
            patches[k] = img[y:y + ph, x:x + pw, :]
            k += 1
    return patches


def extract_matched_random_patches(imgs_a: np.ndarray, imgs_b: np.ndarray, num_patches: int, patch_size=(8, 8), seed: int = 42):
    """
    Extract the SAME random coordinates from two batches of images.

    Parameters
    ----------
    imgs_a, imgs_b : (B, H, W, C)
    num_patches : int
    patch_size : (ph, pw)
    seed : int

    Returns
    -------
    patches_a, patches_b : (B * num_patches, ph, pw, C)
    coords : list[(y, x)] of length num_patches
    """
    assert imgs_a.shape == imgs_b.shape, "Both inputs must have the same shape."
    rng = np.random.default_rng(seed)

    b, h, w, c = imgs_a.shape
    ph, pw = patch_size

    coords = _sample_coords(num_patches, h, w, ph, pw, rng)
    pa = []
    pb = []
    for y, x in coords:
        pa.append(imgs_a[:, y:y + ph, x:x + pw, :])  # (B, ph, pw, C)
        pb.append(imgs_b[:, y:y + ph, x:x + pw, :])  # (B, ph, pw, C)
    patches_a = np.concatenate(pa, axis=0)  # (B*num_patches, ph, pw, C)
    patches_b = np.concatenate(pb, axis=0)  # (B*num_patches, ph, pw, C)
    return patches_a, patches_b, [tuple(rc) for rc in coords]


# -----------------------------
# Reassembly
# -----------------------------
def reassemble_from_patches(similarity_patches: np.ndarray, coords, image_shape, patch_size):
    """
    Reassemble per-patch similarity back into full images with averaging on overlaps.

    Accepts either:
      - (B*num_patches, 1)         -> scalar score per patch (broadcast over ph×pw)
      - (B*num_patches, ph, pw, 1) -> per-pixel map per patch

    Parameters
    ----------
    similarity_patches : np.ndarray
    coords            : list[(y, x)] length == num_patches
    image_shape       : (B, H, W, C)
    patch_size        : (ph, pw)

    Returns
    -------
    (B, H, W, 1) float32
    """
    import numpy as np

    b, h, w, _ = image_shape
    ph, pw = patch_size
    num_patches = len(coords)

    sim = np.asarray(similarity_patches)
    out = np.zeros((b, h, w, 1), dtype=np.float32)
    cnt = np.zeros((b, h, w, 1), dtype=np.float32)

    # Case 1: scalar score per patch -> (B*num_patches, 1)
    if sim.ndim == 2 and sim.shape[1] == 1:
        # reshape to (num_patches, B, 1)
        sim = sim.reshape(num_patches, b, 1).astype(np.float32)
        for idx, (y, x) in enumerate(coords):
            # broadcast (B,1,1,1) over the patch region
            val = sim[idx, :, 0][:, None, None, None]
            out[:, y:y + ph, x:x + pw, :] += val
            cnt[:, y:y + ph, x:x + pw, :] += 1.0

    # Case 2: full patch map per patch -> (B*num_patches, ph, pw, 1)
    elif sim.ndim == 4 and tuple(sim.shape[1:]) == (ph, pw, 1):
        sim = sim.reshape(num_patches, b, ph, pw, 1).astype(np.float32)
        for idx, (y, x) in enumerate(coords):
            out[:, y:y + ph, x:x + pw, :] += sim[idx]
            cnt[:, y:y + ph, x:x + pw, :] += 1.0

    else:
        raise ValueError(
            f"Unexpected similarity_patches shape {sim.shape}; "
            f"expected (B*num_patches, 1) or (B*num_patches, {ph}, {pw}, 1)."
        )

    np.maximum(cnt, 1.0, out=cnt)  # avoid division by zero
    return out / cnt


# -----------------------------
# End-to-end helper
# -----------------------------
import numpy as np
import torch

# assumes these exist and return/accept NumPy in NHWC
# from your_utils_module import extract_matched_random_patches, reassemble_from_patches

import numpy as np
import torch

def process_images_random(
    imgs_a: np.ndarray,
    imgs_b: np.ndarray,
    num_patches: int,
    siamese_model: torch.nn.Module,
    patch_size=(8, 8),
    seed: int = 42,
):
    """
    Extract matched random patches, run Siamese model, and reassemble similarity into full-size maps (PyTorch).

    Parameters
    ----------
    imgs_a, imgs_b : np.ndarray
        Arrays of shape (B, H, W, C) in [-1, 1].
    num_patches : int
        Number of random matched patches to extract.
    siamese_model : torch.nn.Module
        Siamese model returning per-patch similarity maps or scalars.
    patch_size : (int, int)
        Patch size (ph, pw).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        (B, H, W, 1) similarity map.
    """

    from utils import extract_matched_random_patches, reassemble_from_patches

    np.random.seed(seed)
    pa, pb, coords = extract_matched_random_patches(
        imgs_a, imgs_b, num_patches, patch_size, seed=seed
    )

    if pa.size == 0:
        return np.zeros(imgs_a.shape[:3] + (1,), dtype=np.float32)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # NHWC → NCHW
    left  = torch.from_numpy(np.transpose(pa, (0, 3, 1, 2))).float().to(device)
    right = torch.from_numpy(np.transpose(pb, (0, 3, 1, 2))).float().to(device)

    siamese_model.eval()
    with torch.no_grad():
        try:
            sim_t = siamese_model(left, right)
        except TypeError:
            pairs = torch.stack([left, right], dim=1)  # (N, 2, 1, ph, pw)
            sim_t = siamese_model(pairs)

    if sim_t is None:
        raise RuntimeError("Siamese model produced no output.")

    sim_np = sim_t.detach().cpu().numpy()

    # Handle different output shapes
    if sim_np.ndim == 1 or (sim_np.ndim == 2 and sim_np.shape[1] == 1):
        # Scalar similarity per patch
        N = sim_np.shape[0]
        ph, pw = patch_size
        sim_np = np.ones((N, ph, pw, 1), dtype=np.float32) * sim_np.reshape(N, 1, 1, 1)
    elif sim_np.ndim == 4:
        # (N, 1, ph, pw) → (N, ph, pw, 1)
        if sim_np.shape[1] in (1, 2, 3):
            sim_np = np.transpose(sim_np, (0, 2, 3, 1))

        # (N, 1, ph, 1) → expand to (N, ph, ph, 1)
        if sim_np.shape[1:] and sim_np.shape[1] != sim_np.shape[2]:
            if sim_np.shape[1] == 1 and sim_np.shape[3] == 1:
                ph = sim_np.shape[2]
                sim_np = np.tile(sim_np.transpose(0, 2, 1, 3), (1, 1, ph, 1))
    else:
        raise ValueError(f"Unexpected siamese output shape {sim_np.shape}")

    # Ensure single channel
    if sim_np.shape[-1] != 1:
        sim_np = sim_np[..., :1]

    # Reassemble into full-size map
    full_map = reassemble_from_patches(
        sim_np.astype(np.float32),
        coords,
        image_shape=imgs_a.shape,
        patch_size=patch_size,
    )

    return full_map.astype(np.float32)
