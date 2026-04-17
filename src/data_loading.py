from pathlib import Path

import imageio
import numpy as np
import tifffile


def load_video(path: str) -> np.ndarray:
    """Load an AVI video file into a (T, H, W) NumPy array."""

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
    """Load a TIF file into a NumPy array."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIF file not found: {path}")

    data = tifffile.imread(str(path))
    return data.astype(np.float32)


def load_data(path: str) -> np.ndarray:
    """Detect file format and load into a NumPy array."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".avi":
        return load_video(path)
    elif suffix in (".tif", ".tiff"):
        return load_tif(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Supported formats: .avi, .tif, .tiff"
        )


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
