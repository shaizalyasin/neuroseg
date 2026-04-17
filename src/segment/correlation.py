import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


def compute_correlation_map(dff: np.ndarray,
                            window: int = 5,
                            lowpass_sigma: float = 2.0) -> np.ndarray:
    """Build a correlation map from a dF/F stack."""

    T, H, W = dff.shape

    # Step 1: low-pass filter along time axis
    filtered = gaussian_filter1d(dff, sigma=lowpass_sigma, axis=0)

    # Step 2: for each pixel compute mean correlation with neighbors
    pad = window // 2
    corr_map = np.zeros((H, W), dtype=np.float32)

    mean = np.mean(filtered, axis=0)
    std = np.std(filtered, axis=0)
    std[std < 1e-8] = 1e-8

    for dy in range(-pad, pad + 1):
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
                        max_area: int = 500) -> np.ndarray:
    """Threshold a correlation map and label connected components as ROIs."""

    # clean correlation map
    corr_map = np.nan_to_num(corr_map, nan=0.0, posinf=0.0, neginf=0.0)

    if threshold is None:
        if corr_map.max() > corr_map.min():
            threshold = threshold_otsu(corr_map)
        else:
            threshold = 0.0

    binary = corr_map > threshold
    labels, n_found = label(binary)

    # remove too-small and too-large regions
    filtered = np.zeros_like(labels)
    new_id = 0
    for region in regionprops(labels):
        if min_area <= region.area <= max_area:
            new_id += 1
            filtered[labels == region.label] = new_id

    print(f"  Correlation segmentation: {n_found} raw -> {new_id} filtered ROIs "
          f"(threshold={threshold:.3f}, area={min_area}-{max_area})")

    return filtered
