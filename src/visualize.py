import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


def plot_segmentation_overlay(image: np.ndarray,
                              labels: np.ndarray,
                              title: str = "Segmentation overlay",
                              ax: plt.Axes = None,
                              save_path: str = None) -> plt.Axes:
    """Display image in grayscale with colored contours around each ROI."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # background image contrast stretch
    p1, p99 = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)

    n_neurons = labels.max()
    if n_neurons > 0:
        rng = np.random.RandomState(42)
        colors = rng.rand(n_neurons, 3)

        boundaries = find_boundaries(labels, mode="thick")
        overlay = np.zeros((*labels.shape, 4), dtype=np.float32)

        for i in range(1, n_neurons + 1):
            roi_boundary = boundaries & (labels == i)
            color_idx = i - 1
            overlay[roi_boundary, :3] = colors[color_idx]
            overlay[roi_boundary, 3] = 1.0

        ax.imshow(overlay)

    ax.set_title(f"{title} ({n_neurons} neurons)", fontsize=11)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return ax


def plot_traces(traces: np.ndarray,
                n: int = 10,
                title: str = "Activity traces",
                offset_scale: float = None,
                fixed_ylim: tuple = None,
                save_path: str = None) -> plt.Figure:
    """Plot fluorescence traces stacked vertically.

    Parameters
    ----------
    traces : np.ndarray, shape (N, T)
        Activity traces.
    n : int
        Number of traces to plot (top N by order, not sorted).
    title : str
        Plot title.
    offset_scale : float, optional
        Vertical spacing between traces. If None, auto‑computed from traces.
    fixed_ylim : tuple (low, high), optional
        If provided, sets the same y‑axis limits for all subplots (useful for comparison).
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
    """
    n_neurons = min(n, traces.shape[0])
    T = traces.shape[1]

    fig, ax = plt.subplots(figsize=(14, max(4, n_neurons * 0.8)))

    if offset_scale is None:
        offset_scale = np.percentile(np.abs(traces[:n_neurons]), 99) * 2.5

    for i in range(n_neurons):
        trace = traces[i]
        offset = -i * offset_scale
        ax.plot(trace + offset, linewidth=0.6, label=f"Neuron {i+1}")
        ax.text(-T * 0.02, offset, f"N{i+1}", fontsize=8,
                ha="right", va="center", color="gray")

    ax.set_xlabel("Frame")
    ax.set_ylabel("dF/F (offset)")
    ax.set_title(f"{title} (top {n_neurons} neurons)")
    ax.set_xlim(0, T)
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    # Hide top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.tight_layout()
    return fig


def plot_comparison(image: np.ndarray,
                    labels_corr: np.ndarray,
                    labels_cellpose: np.ndarray,
                    traces_corr: np.ndarray = None,
                    traces_cellpose: np.ndarray = None,
                    n_traces: int = 10,
                    fixed_ylim: tuple = None,
                    save_path: str = None) -> plt.Figure:
    """Side‑by‑side comparison of correlation and Cellpose results.

    Parameters
    ----------
    image : np.ndarray
        2D projection for background.
    labels_corr, labels_cellpose : np.ndarray
        Label images.
    traces_corr, traces_cellpose : np.ndarray, optional
        Activity traces (N, T) for each method.
    n_traces : int
        Number of traces to show (if traces provided).
    fixed_ylim : tuple, optional
        Shared y‑axis limits for the trace subplots (makes comparison fair).
    save_path : str, optional
        Save figure to this path.
    """
    has_traces = traces_corr is not None and traces_cellpose is not None
    n_rows = 2 if has_traces else 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 7 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Segmentation overlays
    plot_segmentation_overlay(image, labels_corr,
                              title="Correlation method", ax=axes[0, 0])
    plot_segmentation_overlay(image, labels_cellpose,
                              title="Cellpose method", ax=axes[0, 1])

    # Traces (if provided)
    if has_traces:
        n_corr = min(n_traces, traces_corr.shape[0])
        n_cell = min(n_traces, traces_cellpose.shape[0])
        T = traces_corr.shape[1]

        # Compute offset scales separately, but can also use fixed_ylim
        offset_corr = np.percentile(np.abs(traces_corr[:n_corr]), 99) * 2.5
        offset_cell = np.percentile(np.abs(traces_cellpose[:n_cell]), 99) * 2.5

        for i in range(n_corr):
            axes[1, 0].plot(traces_corr[i] - i * offset_corr, linewidth=0.5)
        axes[1, 0].set_title(f"Correlation traces ({n_corr} neurons)")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_xlim(0, T)
        if fixed_ylim is not None:
            axes[1, 0].set_ylim(fixed_ylim)

        for i in range(n_cell):
            axes[1, 1].plot(traces_cellpose[i] - i * offset_cell, linewidth=0.5)
        axes[1, 1].set_title(f"Cellpose traces ({n_cell} neurons)")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_xlim(0, T)
        if fixed_ylim is not None:
            axes[1, 1].set_ylim(fixed_ylim)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
    