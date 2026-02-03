import cv2
import numpy as np
from pathlib import Path
import json
import PIL
import PIL.Image
import pandas as pd
import fsspec


def load_image_pil(path: "str | Path") -> PIL.Image.Image:
    """
    PIL is needed for reading EXIF metadata (e.g. timestamp) from a JPEG image
    (otherwise use cv2 lib)

    Note: path can be a local file or cloud bucket uri
    """
    with fsspec.open(path, "rb") as f:
        image = PIL.Image.open(f)
        image.load()

    return image


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV failed to decode image bytes")
    return img


def _read_bytes_any(path: "str | Path") -> bytes:
    p = str(path)
    with fsspec.open(p, "rb") as f:
        return f.read()


def load_image(path: "str | Path") -> np.ndarray:
    """
    Read image and return as RGB or RGBA or GRAY depending on image

    Notes:
    - opencv reads/writes as BGR by default, so we explicitly convert to RGB for consistency
    - path can be a local file or cloud bucket uri
    """
    # read raw to keep alpha if present
    image = _decode_image_bytes(_read_bytes_any(path))
    if image is None:
        raise RuntimeError(f"Failed to read image at {path}")

    # grayscale (H, W)
    if image.ndim == 2:
        # Anything to do here or is it read correctly by default?
        return image

    # color images
    ch = image.shape[2]
    if ch == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if ch == 4:
        # Leave the alpha channel as-is if there are 4 channels
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    # unexpected channels: return as-is
    return image


def load_image_mask(mask_path: Path) -> np.ndarray:
    """
    Load an image to be interpreted as a boolean mask.
    Returns an (H, W) boolean array where nonzero pixels => True.

    Logic:
    - If the image is already single-channel (H, W) use it directly.
    - If the image has 4 channels (e.g. RGBA/BGRA) use the alpha channel as
      the mask (alpha > 0 => included).
    - If the image has 3 channels (BGR) convert to grayscale with
      cv2.cvtColor(..., COLOR_BGR2GRAY) and treat nonzero as included.
    """
    # read raw to keep alpha if present
    img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read mask image at {mask_path}")

    # grayscale single-channel (H, W)
    if img.ndim == 2:
        arr = img
    else:
        ch = img.shape[2]
        if ch == 4:
            # BGRA -> use alpha channel as mask (last channel)
            arr = img[:, :, 3]
        elif ch == 3:
            # BGR -> convert to grayscale (uint8)
            arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # unexpected channel count: reduce to grayscale via mean as fallback
            # This covers exotic files with >4 channels; compute per-pixel mean across channels
            arr = np.mean(img, axis=2).astype(img.dtype)

    # Return boolean mask: True where mask pixel is nonzero
    return arr != 0


def write_image(image: np.ndarray, path: Path):
    """
    Write image in RGB/RGBA (or grayscale).
    Converts RGB->BGR, RGBA->BGRA as needed.
    Handles float images by scaling to uint8.
    """

    if image.ndim == 2:
        out = image  # grayscale
    elif image.shape[2] == 3:
        out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[2] == 4:
        out = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError("Unsupported number of channels: %s" % (image.shape,))
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise RuntimeError(f"Failed to write output image to {path}")


def read_image_table(path: Path) -> pd.DataFrame:
    """
    Reads CSV or Parquet containing image_path,timestamp.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"--images must be a .csv or .parquet file, got: {path}")

    if "image_path" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("Input must contain columns: image_path, timestamp")

    # Drop rows with missing image_path; keep timestamp as string
    df = df.dropna(subset=["image_path"]).copy()
    df["timestamp"] = df["timestamp"].astype(str)

    return df


def load_polygons_from_json(json_path: Path) -> list[np.ndarray]:
    """
    Load polygons from JSON.
    Expected JSON format (from the R export):
      [
        [[x1, y1], [x2, y2], ...],          # polygon 1
        [[x1, y1], [x2, y2], ...],          # polygon 2
        ...
      ]
    Returns: list of polygons where each polygon is an array of coordinates
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    polygons = [np.asarray(poly, dtype=np.float32) for poly in data]
    return polygons
