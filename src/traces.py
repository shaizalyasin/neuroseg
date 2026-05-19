import numpy as np


def extract_traces(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Extract mean fluorescence trace for each neuron ROI.

    Parameters
    ----------
    stack : np.ndarray
        dF/F or raw stack of shape (T, H, W).
    labels : np.ndarray
        Integer label image of shape (H, W) where 0 = background.

    Returns
    -------
    np.ndarray
        Traces array of shape (N, T) where N is the number of ROIs.
    """

    n_neurons = labels.max()
    T = stack.shape[0]

    traces = np.zeros((n_neurons, T), dtype=np.float32)

    for i in range(1, n_neurons + 1):
        mask = labels == i
        traces[i - 1] = stack[:, mask].mean(axis=1)

    print(f"  Extracted {n_neurons} traces, {T} time points each")
    return traces


def compute_snr(traces: np.ndarray,
                baseline_percentile: float = 20.0,
                noise_fraction: float = 0.2) -> np.ndarray:
    """
    SNR = (peak - baseline) / noise_std
    
    Baseline is the `baseline_percentile` of the entire trace.
    Noise is estimated as the standard deviation of the quietest `noise_fraction`
    of the trace (based on moving window), not just the first frames.
    """
    N, T = traces.shape
    baseline = np.percentile(traces, baseline_percentile, axis=1)
    
    # Find the quietest contiguous segment of length noise_fraction*T
    window = max(1, int(T * noise_fraction))
    noise_std = np.zeros(N)
    for i in range(N):
        # Rolling window standard deviation
        min_std = np.inf
        for start in range(T - window + 1):
            seg = traces[i, start:start+window]
            std_seg = np.std(seg)
            if std_seg < min_std:
                min_std = std_seg
        noise_std[i] = min_std
    noise_std = np.where(noise_std > 1e-6, noise_std, np.nan)
    
    peak = traces.max(axis=1)
    snr = (peak - baseline) / noise_std
    return snr