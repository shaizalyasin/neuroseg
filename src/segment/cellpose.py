import numpy as np
from cellpose.models import CellposeModel

# Per-model-type cache: avoids reloading weights on every call,
# but correctly handles switching between cyto3 and nuclei models.
_model_cache = {}


def _get_model(model_type: str, use_gpu: bool = False) -> CellposeModel:
    """Return a cached CellposeModel, loading it only on first use per type."""
    if model_type not in _model_cache:
        print(f"  Cellpose: loading model '{model_type}' ...")
        _model_cache[model_type] = CellposeModel(model_type=model_type, gpu=use_gpu)
    return _model_cache[model_type]


def segment_cellpose(image_2d: np.ndarray,
                     model_type: str = "cyto3",
                     diameter: float = None,
                     indicator: str = "cytoplasmic",
                     use_gpu: bool = False) -> np.ndarray:
    """Segment neurons using Cellpose on a 2D projection.

    Parameters
    image_2d : np.ndarray
        2D image (H, W), e.g. max or mean projection.
    model_type : str
        Cellpose model name. Overridden to 'nuclei' when indicator is 'nls'.
    diameter : float or None
        Expected neuron diameter in pixels. If None, Cellpose auto-estimates.
    indicator : str
        'cytoplasmic' or 'nls'. When 'nls', forces the 'nuclei' model
        since NLS-GCaMP is localised to the nucleus.
    use_gpu : bool
        Whether to use GPU (default False).

    Returns
    np.ndarray
        Label image (H, W) with 0=background, 1..N=neurons.
    """
    # Override model for nuclear indicators
    if indicator == "nls":
        model_type = "nuclei"
        print(f"  Cellpose: indicator=nls → switching to '{model_type}' model")

    if diameter is None:
        print("  Cellpose: auto-estimating diameter (diameter=None)")
    else:
        print(f"  Cellpose: using fixed diameter = {diameter}")

    print(f"  Cellpose: model={model_type}, indicator={indicator}")

    model = _get_model(model_type, use_gpu=use_gpu)

    masks, _, _ = model.eval(image_2d, diameter=diameter)
    n_cells = masks.max()
    print(f"  Cellpose segmentation: {n_cells} cells detected")

    return masks