import numpy as np
from cellpose.models import CellposeModel


def segment_cellpose(image_2d: np.ndarray,
                     diameter: float = None) -> np.ndarray:
    """Segment neurons using Cellpose."""

    model = CellposeModel(gpu=False)

    masks, flows, styles = model.eval(
        image_2d,
        diameter=diameter,
    )

    n_cells = masks.max()
    print(f"  Cellpose segmentation: {n_cells} cells detected "
          f"(diameter={diameter if diameter else 'auto'})")

    return masks
