import numpy as np


def extract_traces(stack: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Extract mean fluorescence trace for each neuron ROI."""

    n_neurons = labels.max()
    T = stack.shape[0]

    traces = np.zeros((n_neurons, T), dtype=np.float32)

    for i in range(1, n_neurons + 1):
        mask = labels == i
        traces[i - 1] = stack[:, mask].mean(axis=1)

    print(f"  Extracted {n_neurons} traces, {T} time points each")
    return traces
