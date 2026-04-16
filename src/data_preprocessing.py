import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift


def motion_correct(stack: np.ndarray, reference: str = "mean") -> np.ndarray:
    """Align every frame to a reference using phase cross correlation.
    """
    if reference == "mean":
        ref_image = np.mean(stack, axis=0)
    elif reference == "first":
        ref_image = stack[0]
    else:
        raise ValueError(f"Unknown reference type: {reference}")

    corrected = np.empty_like(stack)

    for i in range(stack.shape[0]):
        # shift, error, phase_diff
        detected_shift, _, _ = phase_cross_correlation(
            ref_image, stack[i], upsample_factor=10
        )
        corrected[i] = shift(stack[i], detected_shift, mode="nearest")

    return corrected


def compute_dff(stack: np.ndarray, baseline_percentile: float = 20.0) -> np.ndarray:
    """Compute dF/F — the relative fluorescence change over time.
    """

    f0 = np.percentile(stack, baseline_percentile, axis=0)
    f0_safe = np.where(f0 > 1.0, f0, np.nan)

    dff = (stack - f0) / f0_safe
    dff = np.nan_to_num(dff, nan=0.0)

    return dff


def subtract_background(stack: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Remove slow spatial background using Gaussian blur subtraction.
    """
    result = np.empty_like(stack)

    for i in range(stack.shape[0]):
        background = gaussian_filter(stack[i], sigma=sigma)
        result[i] = stack[i] - background

    return result


def preprocess(stack: np.ndarray,
               do_motion_correct: bool = True,
               do_bg_subtract: bool = True,
               bg_sigma: float = 50.0,
               baseline_percentile: float = 20.0) -> np.ndarray:
    """Full preprocessing pipeline: 
    Motion correction -> dF/F -> Background subtract
    Returns array (T, H, W).
    """
    result = stack.copy()

    if do_motion_correct:
        print("Motion correction ...")
        result = motion_correct(result)

    print("Computing dF/F ...")
    result = compute_dff(result, baseline_percentile=baseline_percentile)

    if do_bg_subtract:
        print("Background subtraction ...")
        result = subtract_background(result, sigma=bg_sigma)

    print("Done.")
    return result
