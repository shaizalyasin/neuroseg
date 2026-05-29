import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, disk


_PALETTE = [
    (0.22, 0.60, 1.00),   # blue
    (0.18, 0.80, 0.44),   # green
    (1.00, 0.40, 0.28),   # red-orange
    (0.80, 0.50, 1.00),   # purple
    (1.00, 0.70, 0.20),   # amber
    (0.20, 0.85, 0.95),   # cyan
    (1.00, 0.30, 0.60),   # pink
    (0.40, 0.90, 0.30),   # lime
    (1.00, 0.55, 0.10),   # orange
    (0.50, 0.30, 1.00),   # violet
    (0.90, 0.85, 0.10),   # yellow
    (0.10, 0.70, 0.55),   # teal
    (1.00, 0.20, 0.20),   # bright red
    (0.30, 0.60, 0.90),   # steel blue
    (0.80, 0.40, 0.10),   # brown-orange
    (0.60, 0.95, 0.60),   # mint
]


def _get_colors(n: int) -> np.ndarray:
    """Return (n, 3) RGB array cycling through the perceptual palette."""
    return np.array([_PALETTE[i % len(_PALETTE)] for i in range(n)])


# 1. Segmentation overlay
def plot_segmentation_overlay(image: np.ndarray,
                              labels: np.ndarray,
                              title: str = "Segmentation overlay",
                              ax: plt.Axes = None,
                              save_path: str = None) -> plt.Axes:
    """Grayscale image with coloured contour borders around each ROI.

    Contours are drawn as thick (3 px) coloured rings — fully opaque —
    so they are clearly visible against any tissue background.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    p1, p99 = np.percentile(image, [1, 99])
    ax.imshow(image, cmap="gray", vmin=p1, vmax=p99)

    n_neurons = int(labels.max())
    if n_neurons > 0:
        colors = _get_colors(n_neurons)

        overlay = np.zeros((*labels.shape, 4), dtype=np.float32)

        selem = disk(1)
        for i in range(1, n_neurons + 1):
            mask     = labels == i
            dilated  = dilation(mask, selem)
            boundary = find_boundaries(dilated, mode="thick")
            col      = colors[i - 1]
            overlay[boundary, :3] = col
            overlay[boundary, 3]  = 1.0

        ax.imshow(overlay, interpolation="nearest")

    ax.set_title(f"{title} ({n_neurons} neurons)", fontsize=11)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return ax


# 2. Fluorescence traces
def plot_traces(traces: np.ndarray,
                n: int = 10,
                title: str = "Activity traces",
                offset_scale: float = None,
                fixed_ylim: tuple = None,
                save_path: str = None) -> plt.Figure:
    """Fluorescence traces stacked vertically on a clean white background.

    Parameters
    traces      : (N, T) dF/F array.
    n           : number of traces to show.
    title       : plot title.
    offset_scale: vertical spacing; auto-computed when None.
    fixed_ylim  : shared y-axis limits (useful for cross-method comparison).
    save_path   : file path to save the figure.
    """
    n_neurons = min(n, traces.shape[0])
    T = traces.shape[1]
    colors = _get_colors(n_neurons)

    fig, ax = plt.subplots(figsize=(14, max(4, n_neurons * 0.85)))

    if offset_scale is None:
        offset_scale = np.percentile(np.abs(traces[:n_neurons]), 99) * 2.5

    for i in range(n_neurons):
        trace  = traces[i]
        offset = -i * offset_scale
        col    = colors[i]
        ax.plot(trace + offset, linewidth=0.8, color=col)
        ax.text(-T * 0.02, offset, f"N{i+1}",
                fontsize=8, ha="right", va="center", color=col, fontweight="bold")

    ax.set_xlabel("Frame")
    ax.set_ylabel("dF/F (offset)")
    ax.set_title(f"{title} (top {n_neurons} neurons)")
    ax.set_xlim(0, T)
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# 3. Side-by-side comparison
def plot_comparison(image: np.ndarray,
                    labels_corr: np.ndarray,
                    labels_cellpose: np.ndarray,
                    traces_corr: np.ndarray = None,
                    traces_cellpose: np.ndarray = None,
                    n_traces: int = 10,
                    fixed_ylim: tuple = None,
                    save_path: str = None) -> plt.Figure:
    """Side-by-side comparison of Correlation vs. Cellpose segmentation.

    Parameters
    image                          : 2D projection for the background.
    labels_corr, labels_cellpose   : integer label images.
    traces_corr, traces_cellpose   : (N, T) dF/F arrays (optional).
    n_traces                       : number of traces to show per method.
    fixed_ylim                     : shared y-axis limits for trace panels.
    save_path                      : file path to save the figure.
    """
    has_traces = traces_corr is not None and traces_cellpose is not None
    n_rows = 2 if has_traces else 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 7 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Row 0: segmentation overlays
    plot_segmentation_overlay(image, labels_corr,
                              title="Correlation method", ax=axes[0, 0])
    plot_segmentation_overlay(image, labels_cellpose,
                              title="Cellpose method",    ax=axes[0, 1])

    # Row 1: traces
    if has_traces:
        n_corr = min(n_traces, traces_corr.shape[0])
        n_cell = min(n_traces, traces_cellpose.shape[0])
        T = traces_corr.shape[1]

        offset_corr = np.percentile(np.abs(traces_corr[:n_corr]), 99) * 2.5
        offset_cell = np.percentile(np.abs(traces_cellpose[:n_cell]), 99) * 2.5
        colors_c = _get_colors(n_corr)
        colors_p = _get_colors(n_cell)

        for i in range(n_corr):
            axes[1, 0].plot(traces_corr[i] - i * offset_corr,
                            linewidth=0.6, color=colors_c[i])
        axes[1, 0].set_title(f"Correlation traces ({n_corr} neurons)")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_xlim(0, T)
        axes[1, 0].spines["top"].set_visible(False)
        axes[1, 0].spines["right"].set_visible(False)
        if fixed_ylim is not None:
            axes[1, 0].set_ylim(fixed_ylim)

        for i in range(n_cell):
            axes[1, 1].plot(traces_cellpose[i] - i * offset_cell,
                            linewidth=0.6, color=colors_p[i])
        axes[1, 1].set_title(f"Cellpose traces ({n_cell} neurons)")
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_xlim(0, T)
        axes[1, 1].spines["top"].set_visible(False)
        axes[1, 1].spines["right"].set_visible(False)
        if fixed_ylim is not None:
            axes[1, 1].set_ylim(fixed_ylim)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig
