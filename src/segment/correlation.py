import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from tqdm import tqdm


def compute_correlation_map(dff: np.ndarray,
                            window: int = 5,
                            lowpass_sigma: float = 2.0) -> np.ndarray:
    """Build a correlation map from a dF/F stack.

    For every pixel, compute the average Pearson correlation with its
    neighbors in a (window x window) patch. Neuron bodies appear as
    bright blobs because their pixels fluctuate in sync.

    Parameters
    dff : np.ndarray
        dF/F stack of shape (T, H, W).
    window : int
        Size of the neighborhood window (must be odd).
    lowpass_sigma : float
        Sigma for temporal low-pass filter before correlation.

    Returns
    np.ndarray
        Correlation map of shape (H, W), values roughly in [-1, 1].
    """

    T, H, W = dff.shape
    n_shifts = (window * window) - 1
    print(f"  Computing correlation map: {H}x{W} image, "
          f"{window}x{window} window ({n_shifts} neighbor shifts)")

    # Step 1: low-pass filter along time axis
    filtered = gaussian_filter1d(dff, sigma=lowpass_sigma, axis=0)

    # Step 2: for each pixel compute mean correlation with neighbors
    pad = window // 2
    corr_map = np.zeros((H, W), dtype=np.float32)

    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0)
    std[std < 1e-8] = 1e-8

    for dy in tqdm(range(-pad, pad + 1), desc="Correlation map"):
        for dx in range(-pad, pad + 1):
            if dy == 0 and dx == 0:
                continue

            # Center region
            y0_c = max(0, -dy)
            y1_c = H - max(0, dy)
            x0_c = max(0, -dx)
            x1_c = W - max(0, dx)

            # Neighbor region
            y0_n = max(0, dy)
            y1_n = H - max(0, -dy)
            x0_n = max(0, dx)
            x1_n = W - max(0, -dx)

            center = filtered[:, y0_c:y1_c, x0_c:x1_c]
            neighbor = filtered[:, y0_n:y1_n, x0_n:x1_n]

            # Pearson correlation
            c_mean = mean[y0_c:y1_c, x0_c:x1_c]
            c_std = std[y0_c:y1_c, x0_c:x1_c]
            n_mean = mean[y0_n:y1_n, x0_n:x1_n]
            n_std = std[y0_n:y1_n, x0_n:x1_n]

            cov = np.mean((center - c_mean) * (neighbor - n_mean), axis=0)
            r = cov / (c_std * n_std)

            corr_map[y0_c:y1_c, x0_c:x1_c] += r

    n_neighbors = (window * window) - 1
    corr_map /= n_neighbors

    return corr_map


def segment_correlation(corr_map: np.ndarray,
                        threshold: float = None,
                        min_area: int = 20,
                        max_area: int = 500,
                        indicator: str = "cytoplasmic") -> np.ndarray:
    """Threshold a correlation map and label connected components as ROIs.

    Parameters
    corr_map : np.ndarray
        Correlation map of shape (H, W).
    threshold : float or None
        Binarization threshold. If None, uses Otsu's method.
    min_area : int
        Minimum ROI area in pixels.
    max_area : int
        Maximum ROI area in pixels.
    indicator : str
        'cytoplasmic' or 'nls'. NLS nuclei are small and round, so
        tighter area and eccentricity filters are applied.

    Returns
    np.ndarray
        Integer label image (H, W) where 0 = background, 1..N = ROI IDs.
    """
    # Override area bounds and shape constraints based on indicator
    if indicator == "nls":
        max_area = min(max_area, 300)
        max_eccentricity = 0.85
        print(f"  NLS mode: max_area={max_area}, max_eccentricity={max_eccentricity}")
    else:
        max_eccentricity = 1.0

    corr_map = np.nan_to_num(corr_map, nan=0.0, posinf=0.0, neginf=0.0)

    if threshold is None:
        if corr_map.max() > corr_map.min():
            threshold = threshold_otsu(corr_map)
        else:
            threshold = 0.0

    binary = corr_map > threshold
    labels, n_found = label(binary)

    # Filter by area and eccentricity
    filtered = np.zeros_like(labels)
    new_id = 0
    n_rejected_area = 0
    n_rejected_shape = 0
    for region in regionprops(labels):
        if not (min_area <= region.area <= max_area):
            n_rejected_area += 1
            continue
        if region.eccentricity > max_eccentricity:
            n_rejected_shape += 1
            continue
        new_id += 1
        filtered[labels == region.label] = new_id

    print(f"  Correlation segmentation: {n_found} raw -> {new_id} filtered ROIs "
          f"(threshold={threshold:.3f}, area={min_area}-{max_area})")
    if n_rejected_area > 0:
        print(f"    Rejected {n_rejected_area} ROIs by area")
    if n_rejected_shape > 0:
        print(f"    Rejected {n_rejected_shape} ROIs by eccentricity (>{max_eccentricity:.2f})")

    return filtered
