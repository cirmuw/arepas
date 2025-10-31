import cv2
import numpy as np
import torch
from tqdm import tqdm
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, PATCH_SIZE, N_PATCHES_PER_IMG, HARD_SAMPLE_PERCENTILE, SEED
import utils

# =========================================================
# Image Loading & Preprocessing
# =========================================================

def load_and_preprocess_images(
    inputs,
    pad_value: float = -1.0,
    target_size: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
):
    """
    Load and preprocess images from file paths or NumPy arrays.

    Steps:
    - If given paths, loads with OpenCV.
    - Normalizes from [0, 255] → [-1, 1].
    - Pads to square with pad_value.
    - Resizes to `target_size`.
    - Converts to grayscale (average across channels).

    Parameters
    ----------
    inputs : list[str] | list[np.ndarray]
        Either a list of file paths or a list of already-loaded images (H, W, C).
    pad_value : float
        Value used to pad images (in [-1, 1] range).
    target_size : tuple[int, int]
        Target size (width, height).

    Returns
    -------
    np.ndarray : (N, H, W, 1) float32 array in [-1, 1]
    """
    imgs = []

    for item in tqdm(inputs, desc="Loading Images"):
        # Load if path, otherwise assume numpy array
        if isinstance(item, str):
            img = cv2.imread(item)
            if img is None:
                continue
        elif isinstance(item, np.ndarray):
            img = item.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(item)}")

        # Normalize to [-1, 1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:  # assume [0,255]
            img = img / 255.0
        img = img / 0.5 - 1.0  # [0,1] → [-1,1]

        h, w = img.shape[:2]

        # Pad to square
        if h != w:
            if h < w:
                diff = w - h
                top, bottom = diff // 2, diff - diff // 2
                img = cv2.copyMakeBorder(
                    img, top, bottom, 0, 0,
                    cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
                )
            else:
                diff = h - w
                left, right = diff // 2, diff - diff // 2
                img = cv2.copyMakeBorder(
                    img, 0, 0, left, right,
                    cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
                )

        # Resize and convert to grayscale
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        if img.ndim == 3 and img.shape[2] == 3:
            img = img.mean(axis=2)  # to grayscale
        elif img.ndim == 2:
            pass  # already grayscale
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        imgs.append(img[:,:,np.newaxis])

    return np.asarray(imgs, dtype=np.float32)


# =========================================================
# Edge Extraction
# =========================================================

def compute_edges(gray_image: np.ndarray) -> np.ndarray:
    """
    Compute an edge map using Canny with Otsu-derived thresholds.

    Parameters
    ----------
    gray_image : (H, W) or (H, W, 1) float32 in [-1, 1]

    Returns
    -------
    (H, W) float32 in [0, 255] (uint8-like) as float32
    """
    if gray_image.ndim == 3:
        gray = gray_image[..., 0]
    else:
        gray = gray_image

    # back to [0, 255] uint8
    img_u8 = np.clip((gray + 1.0) * 127.5, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(img_u8, (5, 5), 1)

    # Otsu threshold -> pick high, set low as a ratio
    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = float(otsu) * 0.66
    low = high  # same as in original code; adjust if desired (e.g., 0.5 * high)

    edges = cv2.Canny(blurred, int(low), int(high))
    return edges.astype(np.float32)


# =========================================================
# Data Augmentation
# =========================================================

def _smooth_blob_mask(h: int, w: int, thr: float = 0.5) -> np.ndarray:
    """
    Create a smooth, irregular binary blob mask of shape (h, w) with {0,1}.
    """
    noise = np.random.normal(0.5, 0.15, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (15, 15), 0)
    _, mask = cv2.threshold(noise, thr, 1.0, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((h, w), dtype=np.uint8)

    largest = max(cnts, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(largest, True)
    smooth = cv2.approxPolyDP(largest, eps, True)

    out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out, [smooth], -1, 1, thickness=-1)
    return out

def random_patch_swap(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Swap a smooth, irregularly-shaped patch within the image.

    Parameters
    ----------
    image : (H, W) or (H, W, 1) float32
    rng   : optional numpy Generator for reproducibility

    Returns
    -------
    image_aug : same shape as input
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.copy()
    h, w = img.shape[:2]

    ph = int(rng.integers(max(1, h // 100), max(2, h // 3 + 1)))
    pw = int(rng.integers(max(1, w // 100), max(2, w // 3 + 1)))

    y1 = int(rng.integers(0, h - ph + 1))
    x1 = int(rng.integers(0, w - pw + 1))
    patch = img[y1:y1 + ph, x1:x1 + pw]

    mask = _smooth_blob_mask(ph, pw).astype(bool)

    # find a sufficiently different target position
    for _ in range(10):
        y2 = int(rng.integers(0, h - ph + 1))
        x2 = int(rng.integers(0, w - pw + 1))
        if abs(y1 - y2) > ph // 2 or abs(x1 - x2) > pw // 2:
            break

    target = img[y2:y2 + ph, x2:x2 + pw]
    np.copyto(target, patch, where=mask)
    img[y2:y2 + ph, x2:x2 + pw] = target
    return img


# =========================================================
# Dataset Preparation
# =========================================================

def prepare_dataset(
    images: np.ndarray,
    augment: bool = False,
    augmentations_per_image: int = 3,
    swaps_per_augmentation: int = 20,
    rng: np.random.Generator | None = None,
):
    """
    Build (X, Y) pairs where X is a (possibly augmented) skeleton and Y is the original image.

    Parameters
    ----------
    images : (N, H, W, 1) float32 in [-1, 1]
    augment : if True, apply irregular patch swaps to the skeleton
    augmentations_per_image : how many augmented versions per image
    swaps_per_augmentation  : how many swaps to apply per augmented version
    rng : optional numpy Generator for reproducibility

    Returns
    -------
    X : (M, H, W, 1) float32 in [-1, 1]
    Y : (M, H, W, 1) float32 in [-1, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    X, Y = [], []
    for img in tqdm(images, desc="Preparing Dataset"):
        # skeleton from grayscale
        edges = compute_edges(img) / 255.0  # [0,1]
        edges = (edges * 2.0 - 1.0).astype(np.float32)        # [-1,1]

        if augment:
            for _ in range(augmentations_per_image):
                aug = edges.copy()
                for _ in range(swaps_per_augmentation):
                    aug = random_patch_swap(aug, rng=rng)
                X.append(aug[:, :, None])
                Y.append(img)
        else:
            X.append(edges[:, :, None])
            Y.append(img)

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

def build_hard_example_pairs(
    generator: torch.nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    patch_size: int = PATCH_SIZE,
    patches_per_image: int = N_PATCHES_PER_IMG,
    hard_percentile: float = HARD_SAMPLE_PERCENTILE,
    seed: int = SEED,
):
    """
    Build positive/negative patch pairs for Siamese training using hard-example mining (PyTorch).

    Steps:
      1. Generate model predictions for a subsampled training set.
      2. Extract random patches from predictions and corresponding ground truth.
      3. Compute per-patch MSE and select top hard_percentile patches (highest errors).
      4. Form positive (GT, pred) and negative (shuffled GT, same pred) pairs.
      5. Return pairs as a single array shaped (N, 2, 1, patch, patch) and labels (N, 1).

    Parameters
    ----------
    generator : torch.nn.Module
        PyTorch generator (expects NCHW tensor in [-1, 1], returns NCHW in [-1, 1]).
    X_train, Y_train : np.ndarray
        Arrays of shape (N, H, W, C) in [-1, 1].
    patch_size : int
        Square patch size.
    patches_per_image : int
        Number of random patches per image.
    hard_percentile : float
        Percentile cutoff (e.g., 90 -> use top 10% hardest patches).
    seed : int
        Random seed.

    Returns
    -------
    X_pairs : np.ndarray
        Shape (N_pairs*2, 2, 1, patch, patch) where channel=1. First axis stacks positives then negatives.
    y_pairs : np.ndarray
        Shape (N_pairs*2, 1). 0 = similar (positive), 1 = dissimilar (negative).
    """
    # -----------------------------
    # 1) Predictions on a subsample
    # -----------------------------
    # Subsample to reduce cost (every 3rd image to match TF code)
    X_sub = np.asarray(X_train)[::3]
    Y_sub = np.asarray(Y_train)[::3]

    # Convert NHWC -> NCHW torch tensor
    x_t = torch.from_numpy(np.transpose(X_sub, (0, 3, 1, 2))).float()

    # Choose device consistent with your training
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    generator.eval()
    preds_list = []
    batch_size = 4  # adjust for your VRAM (try 1–8)
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            batch = x_t[i:i+batch_size].to(device)
            out = generator(batch)
            preds_list.append(out.detach().cpu())
    preds_t = torch.cat(preds_list, dim=0)
    preds = preds_t.numpy()  # N,1,H,W

    # Back to NHWC for patch extraction (utils expects NHWC)
    preds_nhwc = np.transpose(preds, (0, 2, 3, 1))  # N,H,W,1

    # ----------------------------------------------
    # 2) Extract random patches for preds and GT
    # ----------------------------------------------
    pred_patches = utils.extract_patches_random(preds_nhwc, patch_size, patches_per_image)  # (M, p, p, 1)
    gt_patches   = utils.extract_patches_random(Y_sub,       patch_size, patches_per_image)  # (M, p, p, 1)

    # -------------------------------------------------
    # 3) Compute per-patch MSE and select hardest ones
    # -------------------------------------------------
    mse = ((gt_patches - pred_patches) ** 2).reshape(len(gt_patches), -1).mean(axis=1)
    thr = np.percentile(mse, hard_percentile)
    print(f"Hard example threshold: {thr:.6f}")
    hard_idx = np.flatnonzero(mse > thr)

    # -------------------------------------------------
    # 4) Form positive and negative pairs
    # -------------------------------------------------
    pos_a = gt_patches[hard_idx]   # GT
    pos_b = pred_patches[hard_idx] # Pred

    # negatives: shuffle GT against the same predictions
    rng = np.random.default_rng(seed)
    neg_a = pos_a.copy()
    rng.shuffle(neg_a)

    # Stack positives then negatives
    left  = np.concatenate([pos_a, neg_a], axis=0)  # (2K, p, p, 1)
    right = np.concatenate([pos_b, pos_b], axis=0)  # (2K, p, p, 1)

    # Labels: positives (0), negatives (1)
    y_pairs = np.concatenate(
        [np.zeros((len(hard_idx), 1), dtype=np.float32),
         np.ones((len(hard_idx), 1), dtype=np.float32)],
        axis=0
    )

    # -------------------------------------------------
    # 5) Return as (N, 2, 1, p, p) for PyTorch
    # -------------------------------------------------
    # NHWC -> NCHW (1, p, p)
    left_nchw  = np.transpose(left,  (0, 3, 1, 2))  # (2K, 1, p, p)
    right_nchw = np.transpose(right, (0, 3, 1, 2))  # (2K, 1, p, p)

    # Combine into (2K, 2, 1, p, p)
    X_pairs = np.stack([left_nchw, right_nchw], axis=1).astype(np.float32)

    return X_pairs, y_pairs
