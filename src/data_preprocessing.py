import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.registration import phase_cross_correlation
from tqdm import tqdm


def motion_correct(stack: np.ndarray, reference: str = "mean") -> np.ndarray:
    """Align every frame to a reference using phase cross correlation.

    Parameters
    stack : np.ndarray
        Input array of shape (T, H, W).
    reference : str
        How to compute the reference image: 'mean' or 'first'.

    Returns
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
    stack : np.ndarray
        Input array of shape (T, H, W).
    bin_size : int
        Number of frames to average together.

    Returns
    np.ndarray
        Binned stack of shape (T // bin_size, H, W).
    """
    T, H, W = stack.shape
    n_bins = T // bin_size
    trimmed = stack[:n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size, H, W).mean(axis=1)
    print(f"  Temporal binning: {T} frames -> {n_bins} frames "
          f"(bin_size={bin_size})")
    return binned.astype(np.float32)


def subtract_background(stack: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Remove slow spatial background using Gaussian blur subtraction.

    Parameters
    stack : np.ndarray
        Input array of shape (T, H, W).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    np.ndarray
        Background-subtracted stack.
    """

    result = np.empty_like(stack)

    for i in tqdm(range(stack.shape[0]), desc="Background subtraction"):
        background = gaussian_filter(stack[i], sigma=sigma)
        result[i] = stack[i] - background

    return result


def compute_dff(stack, fps, window_seconds=30.0, percentile=10.0,
                min_baseline_factor=0.01):
    """dF/F with robust baseline estimation.

    Parameters
    stack : np.ndarray, shape (T, H, W)
        Input fluorescence stack.
    fps : float
        Frame rate in Hz. Required to convert window_seconds to frames.
    window_seconds : float
        Sliding window size in seconds for baseline percentile.
    percentile : float
        Percentile used for baseline estimation (default 10th).
    min_baseline_factor : float
        Fraction of the global 99th percentile used as minimum baseline.
        Prevents division by near-zero values.
    """
    import warnings

    if fps is None:
        raise ValueError(
            "fps must be specified — cannot compute baseline window "
            "without knowing frame rate"
        )

    T, H, W = stack.shape
    window_frames = int(window_seconds * fps)

    if window_frames < 2:
        window_frames = 2
        warnings.warn(f"Baseline window rounded up to 2 frames "
                      f"(fps={fps}, window_seconds={window_seconds})")

    if window_frames > T // 2:
        warnings.warn(
            f"Baseline window ({window_frames} frames = {window_seconds}s) "
            f"is more than half the recording ({T} frames = {T/fps:.1f}s). "
            f"Consider reducing --dff-window-sec."
        )

    print(f"  dF/F: fps={fps}, window={window_seconds}s = {window_frames} frames, "
          f"percentile={percentile}")

    half = window_frames // 2
    dff = np.zeros_like(stack, dtype=np.float32)

    global_99 = np.percentile(stack, 99)
    min_baseline = global_99 * min_baseline_factor

    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        baseline = np.percentile(stack[start:end], percentile, axis=0)
        baseline = np.maximum(baseline, min_baseline)
        dff[t] = (stack[t] - baseline) / baseline

    dff = np.clip(dff, -1.0, 20.0)
    return dff


def preprocess(stack,
               fps: float = None,
               do_motion_correct: bool = True,
               temporal_bin_size: int = None,
               dff_window_sec: float = 30.0,
               edge_crop: int = 5) -> np.ndarray:
    """Minimal preprocessing: motion correct, bin, dF/F, crop.

    Parameters
    stack : np.ndarray, shape (T, H, W)
    fps : float
        Frame rate in Hz. Required for dF/F baseline window.
    do_motion_correct : bool
    temporal_bin_size : int or None
    dff_window_sec : float
        Baseline window in seconds (default 30).
    edge_crop : int
        Pixels to zero at each edge.
    """
    result = stack.copy()

    if do_motion_correct:
        print("Motion correction ...")
        result = motion_correct(result)

    if temporal_bin_size and temporal_bin_size > 1:
        print(f"Temporal binning (size={temporal_bin_size}) ...")
        result = temporal_bin(result, bin_size=temporal_bin_size)
        if fps is not None:
            fps = fps / temporal_bin_size
            print(f"  Effective fps after binning: {fps:.2f}")

    print("Computing dF/F ...")
    result = compute_dff(result, fps=fps, window_seconds=dff_window_sec)

    if edge_crop > 0:
        print(f"Edge cropping ({edge_crop} px) ...")
        result[:, :edge_crop, :] = 0
        result[:, -edge_crop:, :] = 0
        result[:, :, :edge_crop] = 0
        result[:, :, -edge_crop:] = 0

    return result
