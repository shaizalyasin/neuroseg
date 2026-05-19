import numpy as np
from cellpose.models import CellposeModel

_model = None

def get_cellpose_model(use_gpu=False):
    global _model
    if _model is None:
        _model = CellposeModel(gpu=use_gpu)
    return _model

def segment_cellpose(image_2d: np.ndarray,
                     model_type: str = "cyto3",
                     diameter: float = None,
                     indicator: str = None,
                     use_gpu: bool = False) -> np.ndarray:
    """Segment neurons using Cellpose on a 2D projection.

    Parameters
    ----------
    image_2d : np.ndarray
        2D image (H, W), e.g. max projection.
    model_type : str
        Cellpose model name (for logging only; v4+ uses unified model).
    diameter : float or None
        Expected neuron diameter in pixels. If None, Cellpose auto‑estimates.
    indicator : str, optional
        'cytoplasmic' or 'nls' – only for logging, does not affect diameter.
    use_gpu : bool
        Whether to use GPU (default False).

    Returns
    -------
    np.ndarray
        Label image (H, W) with 0=background, 1..N=neurons.
    """
    if diameter is None:
        print("  Cellpose: auto‑estimating diameter (diameter=None)")
    else:
        print(f"  Cellpose: using fixed diameter = {diameter}")

    if indicator:
        print(f"  Cellpose: indicator = {indicator} (for logging only)")

    model = get_cellpose_model(use_gpu=use_gpu)

    masks, _, _ = model.eval(image_2d, diameter=diameter)
    n_cells = masks.max()
    print(f"  Cellpose segmentation: {n_cells} cells detected")

    return masks