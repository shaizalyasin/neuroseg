import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


# def detect_bad_frames(stack: np.ndarray, threshold: float = 3.0) -> list:
#     """Detect frames that deviate strongly from their temporal neighbors.

#     For each frame, computes the mean absolute difference from the rolling
#     mean of +-5 surrounding frames. Frames whose difference exceeds
#     `threshold` standard deviations above the median difference are flagged.

#     Parameters
#     ----------
#     stack : np.ndarray
#         Input array of shape (T, H, W).
#     threshold : float
#         Number of standard deviations above median to flag a frame.

#     Returns
#     -------
#     list
#         Indices of bad frames.
#     """
#     T = stack.shape[0]
#     half_win = 5
#     diffs = np.zeros(T, dtype=np.float32)

#     for i in range(T):
#         lo = max(0, i - half_win)
#         hi = min(T, i + half_win + 1)
#         # Exclude current frame from the local mean
#         neighbors = np.concatenate([stack[lo:i], stack[i+1:hi]], axis=0)
#         if len(neighbors) == 0:
#             continue
#         local_mean = neighbors.mean(axis=0)
#         diffs[i] = np.mean(np.abs(stack[i] - local_mean))

#     median_diff = np.median(diffs)
#     std_diff = np.std(diffs)

#     if std_diff < 1e-8:
#         return []

#     bad = np.where(diffs > median_diff + threshold * std_diff)[0].tolist()
#     return bad


# def remove_bad_frames(stack: np.ndarray, bad_indices: list) -> np.ndarray:
#     """Remove flagged frames from the stack.

#     Parameters
#     ----------
#     stack : np.ndarray
#         Input array of shape (T, H, W).
#     bad_indices : list
#         List of frame indices to remove.

#     Returns
#     -------
#     np.ndarray
#         Stack with bad frames removed, shape (T - len(bad_indices), H, W).
#     """
#     if len(bad_indices) == 0:
#         print("  No bad frames to remove.")
#         return stack

#     total = stack.shape[0]
#     pct = 100.0 * len(bad_indices) / total
#     print(f"  Removing {len(bad_indices)} bad frames "
#           f"({pct:.1f}% of {total} total)")

#     good = np.delete(stack, bad_indices, axis=0)
#     return good


def motion_correct(stack: np.ndarray, reference: str = "mean") -> np.ndarray:
    """Align every frame to a reference using phase cross correlation.

    Parameters
    ----------
    stack : np.ndarray
        Input array of shape (T, H, W).
    reference : str
        How to compute the reference image: 'mean' or 'first'.

    Returns
    -------
    np.ndarray
        Motion-corrected stack of shape (T, H, W).
    """

    if reference == "mean":
        ref_image = np.mean(stack, axis=0)
    elif reference == "first":
        ref_image = stack[0]
    else:
        raise ValueError(f"Unknown reference type: {reference}")

    corrected = np.empty_like(stack)

    for i in tqdm(range(stack.shape[0]), desc="Motion correction"):
        # shift, error, phase_diff
        detected_shift, _, _ = phase_cross_correlation(
            ref_image, stack[i], upsample_factor=10
        )
        corrected[i] = shift(stack[i], detected_shift, mode="nearest")

    return corrected


def temporal_bin(stack: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """Average every `bin_size` consecutive frames into one.

    Useful for oversampled data (e.g. 200 Hz AVI) to reduce noise
    and bring effective frame rate to a more manageable level.

    Parameters
    ----------
    stack : np.ndarray
        Input array of shape (T, H, W).
    bin_size : int
        Number of frames to average together.

    Returns
    -------
    np.ndarray
        Binned stack of shape (T // bin_size, H, W).
    """
    T, H, W = stack.shape
    n_bins = T // bin_size
    # Trim to exact multiple of bin_size
    trimmed = stack[:n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size, H, W).mean(axis=1)
    print(f"  Temporal binning: {T} frames -> {n_bins} frames "
          f"(bin_size={bin_size})")
    return binned.astype(np.float32)


def subtract_background(stack: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Remove slow spatial background using Gaussian blur subtraction.

    Parameters
    ----------
    stack : np.ndarray
        Input array of shape (T, H, W).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Background-subtracted stack.
    """

    result = np.empty_like(stack)

    for i in tqdm(range(stack.shape[0]), desc="Background subtraction"):
        background = gaussian_filter(stack[i], sigma=sigma)
        result[i] = stack[i] - background

    return result


def compute_dff(stack, window_size=30, min_baseline_factor=0.01):
    """dF/F with robust baseline estimation.
    
    Parameters
    ----------
    stack : np.ndarray, shape (T, H, W)
        Input fluorescence stack.
    window_size : int
        Sliding window size (frames) for baseline percentile.
    min_baseline_factor : float
        Fraction of the global 99th percentile used as minimum baseline.
        Prevents division by near‑zero values.
    """
    T, H, W = stack.shape
    dff = np.zeros_like(stack, dtype=np.float32)
    half = window_size // 2

    # Global estimate of typical signal level (99th percentile)
    global_99 = np.percentile(stack, 99)
    min_baseline = global_99 * min_baseline_factor  # e.g., 1% of max typical intensity

    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        baseline = np.percentile(stack[start:end], 10, axis=0)
        # Apply the floor: any baseline below min_baseline is set to min_baseline
        baseline = np.maximum(baseline, min_baseline)
        dff[t] = (stack[t] - baseline) / baseline

    # Optional: clip dF/F to a reasonable range (e.g., -1 to 20)
    dff = np.clip(dff, -1.0, 20.0)
    return dff


def preprocess(stack,
               do_motion_correct: bool = True,
               temporal_bin_size: int = None,
            #    bg_sigma: float = 50.0,
               dff_window: int = 30,
               edge_crop: int = 5) -> np.ndarray:
    """Minimal preprocessing: motion correct (optional), bin, background subtract, dF/F, crop."""
    result = stack.copy()

    if do_motion_correct:
        print("Motion correction ...")
        result = motion_correct(result)

    if temporal_bin_size and temporal_bin_size > 1:
        print(f"Temporal binning (size={temporal_bin_size}) ...")
        result = temporal_bin(result, bin_size=temporal_bin_size)

    # print("Background subtraction ...")
    # result = subtract_background(result, sigma=bg_sigma)
    print("Computing dF/F ...")
    result = compute_dff(result, window_size=dff_window)   # pass the window size

    if edge_crop > 0:
        print(f"Edge cropping ({edge_crop} px) ...")
        result[:, :edge_crop, :] = 0
        result[:, -edge_crop:, :] = 0
        result[:, :, :edge_crop] = 0
        result[:, :, -edge_crop:] = 0

    return result
