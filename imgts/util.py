import numpy as np
import cv2


def polygons_to_image(
    polygons: list[
        np.ndarray
    ],  # polygon Mask as JSON list of polygons (each polygon is a list of coords)
    ref_image: np.ndarray,  # Reference JPEG as ndarray for bounds
):
    height, width = ref_image.shape[:2]

    # Convert polygons to HxW boolean matrix where interior of polygons correspond to
    # True cells
    return create_polygon_mask(polygons, height, width)


def convert_spatial_to_image_coords(
    polygons: list[np.ndarray], height: int
) -> list[np.ndarray]:
    """
    Convert polygons from SpatialPolygons-style coordinates (origin at bottom-left,
    y increasing upwards) to image coordinates (origin at top-left, y increasing downwards).
    i.e.:
    y_img = (height - 1) - y_spatial

    polygons: list of polygons where each polygon is an array of coordinates and the y-coords
    have bottom-left origin

    Returns: list of polygons where y-coords have top-left origin
    """
    converted = []
    for poly in polygons:
        poly_img = poly.copy()
        poly_img[:, 1] = (height - 1) - poly_img[:, 1]
        converted.append(poly_img)
    return converted


def create_polygon_mask(
    polygons: list[np.ndarray], height: int, width: int
) -> np.ndarray:
    """
    Create a binary polygon mask using OpenCV.
    polygons: list of polygons where each polygon is an array of coordinates
    Returns: mask with shape (H, W), dtype uint8, values {0,1}.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for poly in polygons:
        # Round and cast to int32 for OpenCV
        poly_int = np.round(poly).astype(np.int32)
        # OpenCV expects a list of polygons; each polygon is (num_points, 1, 2)
        poly_cv = poly_int.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly_cv], 1)

    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image
    image: (H, W, 3)
    mask: (H, W) with values {0, 1} (or 0/255, or bool)
    Returns: masked image as (H, W, 3)
    """

    if not image.shape[:2] == mask.shape[:2]:
        raise RuntimeError(f"image h*w {image.shape[:2]} != mask h*w {mask.shape[:2]}")

    # Ensure mask is broadcastable: (H, W) -> (H, W, 1)
    mask_3 = mask[..., np.newaxis]

    # Elementwise multiply (broadcast over channels)
    masked_image = image * mask_3

    return masked_image
