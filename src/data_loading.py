from pathlib import Path

import imageio
import numpy as np
import tifffile


def load_video(path: str) -> np.ndarray:
    """Load an AVI video file into a (T, H, W) NumPy array.

    Parameters
    path : str
        Path to the AVI file.

    Returns
    np.ndarray
        Float32 array of shape (T, H, W).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    reader = imageio.get_reader(str(path))
    frames = []
    for frame in reader:
        # Convert RGB → grayscale
        if frame.ndim == 3:
            frame = np.mean(frame, axis=-1)
        frames.append(frame)
    reader.close()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {path}")

    stack = np.array(frames, dtype=np.float32)
    return stack


def load_tif(path: str) -> np.ndarray:
    """Load a TIF file into a NumPy array.

    Parameters
    path : str
        Path to the TIF/TIFF file.

    Returns
    np.ndarray
        Float32 array, typically shape (T, H, W).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIF file not found: {path}")

    with tifffile.TiffFile(str(path)) as tiff:
        data = np.stack([page.asarray() for page in tiff.pages])

    return data.astype(np.float32)


def load_czi(path: str) -> np.ndarray:
    """Load a CZI or .sec file into a (T, H, W) NumPy array.

    Handles multi channel, and other extra dimensions by
    taking the first element along those axes.

    Parameters
    path : str
        Path to the CZI or .sec file.

    Returns
    np.ndarray
        Float32 array of shape (T, H, W).
    """
    from aicspylibczi import CziFile

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CZI file not found: {path}")

    czi = CziFile(str(path))
    dims = czi.get_dims_shape()
    print(f"  CZI dimensions: {dims}")

    data, dim_order = czi.read_image()
    # dim_order is list of tuples like [('T', 764), ('Y', 512), ('X', 248), ('C', 1), ('Z', 1)]
    axes = [d[0] for d in dim_order]
    print(f"  Raw shape: {data.shape}, axes: {axes}")

    # We want to keep only T, Y, X. For all other axes, take index 0.
    target_axes = ['T', 'Y', 'X']
    keep_indices = []
    for ax in target_axes:
        if ax in axes:
            keep_indices.append(axes.index(ax))
        else:
            raise ValueError(f"CZI missing required axis '{ax}'. Found axes: {axes}")

    # Reduce extra axes (all except T,Y,X) by taking first element
    reduce_axes = [i for i, ax in enumerate(axes) if ax not in target_axes]
    for idx in sorted(reduce_axes, reverse=True):
        data = np.take(data, 0, axis=idx)
        axes.pop(idx)

    # Now axes should be exactly ['T','Y','X'] in some order. Permute to T,Y,X.
    if axes != target_axes:
        perm = [axes.index(ax) for ax in target_axes]
        data = np.transpose(data, perm)

    data = data.astype(np.float32)
    print(f"  Extracted: {data.shape[0]} frames, {data.shape[1]}x{data.shape[2]} px")
    return data


def load_data(path: str) -> np.ndarray:
    """Detect file format and load into a NumPy array.

    Supported formats: .avi, .tif/.tiff, .czi, .sec

    Parameters
    path : str
        Path to the data file.

    Returns
    np.ndarray
        Float32 array, typically shape (T, H, W).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".avi":
        return load_video(path)
    elif suffix in (".tif", ".tiff"):
        return load_tif(path)
    elif suffix in (".czi", ".sec"):
        return load_czi(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported formats: .avi, .tif, .tiff, .czi, .sec"
        )


def extract_fps(path: str) -> float:
    """Extract frame rate from file metadata.
    
    Returns
    float or None
        The detected frame rate in Hz, or None if not found.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(str(path)) as tif:
                if tif.imagej_metadata:
                    md = tif.imagej_metadata
                    if 'finterval' in md and md['finterval'] > 0:
                        return 1.0 / md['finterval']
                    elif 'fps' in md:
                        return float(md['fps'])
        except Exception:
            pass

    elif suffix == ".avi":
        try:
            import imageio
            reader = imageio.get_reader(str(path))
            meta = reader.get_meta_data()
            reader.close()
            if 'fps' in meta and meta['fps'] > 0:
                return float(meta['fps'])
        except Exception:
            pass
    
    return None


def summarise(data: np.ndarray, name: str = "") -> None:
    """Print a summary of a loaded array (shape, dtype, value range)."""
    header = f"- {name} -" if name else "- summary -"
    print(header)
    print(f"  shape : {data.shape}")
    print(f"  dtype : {data.dtype}")
    print(f"  min   : {data.min():.4f}")
    print(f"  max   : {data.max():.4f}")
    p1, p99 = np.percentile(data, [1, 99])
    print(f"  p1    : {p1:.4f}")
    print(f"  p99   : {p99:.4f}")
    