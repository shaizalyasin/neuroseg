import numpy as np
import matplotlib.pyplot as plt


def plot_segmentation_overlay(image: np.ndarray,
                              labels: np.ndarray,
                              title: str = "Segmentation overlay",
                              ax: plt.Axes = None,
                              save_path: str = None) -> plt.Axes:
    """The image is displayed in grayscale with colored contours
    drawn around each labeled ROI."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # background image
    p1, p99 = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)

    # colored overlay for ROI boundaries
    n_neurons = labels.max()
    if n_neurons > 0:
        rng = np.random.RandomState(42)
        colors = rng.rand(n_neurons, 3)

        from skimage.segmentation import find_boundaries
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
                save_path: str = None) -> plt.Figure:
    """Plot fluorescence traces stacked vertically."""

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
                    save_path: str = None) -> plt.Figure:
    """Side-by-side comparison of both segmentation methods."""

    has_traces = traces_corr is not None and traces_cellpose is not None
    n_rows = 2 if has_traces else 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 7 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    plot_segmentation_overlay(image, labels_corr,
                              title="Correlation method", ax=axes[0, 0])
    plot_segmentation_overlay(image, labels_cellpose,
                              title="Cellpose method", ax=axes[0, 1])

    if has_traces:
        n_corr = min(n_traces, traces_corr.shape[0])
        n_cell = min(n_traces, traces_cellpose.shape[0])
        T = traces_corr.shape[1]

        offset_corr = np.percentile(np.abs(traces_corr[:n_corr]), 99) * 2.5
        offset_cell = np.percentile(np.abs(traces_cellpose[:n_cell]), 99) * 2.5

        for i in range(n_corr):
            axes[1, 0].plot(traces_corr[i] - i * offset_corr, linewidth=0.5)
        axes[1, 0].set_title(f"Correlation traces ({n_corr} neurons)")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_xlim(0, T)

        for i in range(n_cell):
            axes[1, 1].plot(traces_cellpose[i] - i * offset_cell, linewidth=0.5)
        axes[1, 1].set_title(f"Cellpose traces ({n_cell} neurons)")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_xlim(0, T)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
