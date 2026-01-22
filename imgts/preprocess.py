from . import util
from . import io
from pathlib import Path
from datetime import datetime
import PIL
import PIL.Image
import PIL.ExifTags
import pandas as pd
import argparse


# ---------------------------
# EXIF helper (DateTime)
# ---------------------------
TAGS = {v: k for k, v in PIL.ExifTags.TAGS.items()}
DT_TAG = TAGS.get("DateTime", None)


def get_timestamp(image: PIL.Image.Image) -> datetime | None:
    """Read EXIF DateTime from a JPEG and return ISO 8601 string, or None."""
    exif = image.getexif()
    if not exif or DT_TAG not in exif:
        return None

    raw = exif[DT_TAG]  # e.g. "2025:01:31 12:34:56"
    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")


def images_to_timeseries(jpeg_image_paths: list[str]) -> pd.DataFrame:
    """
    Create a DataFrame timeseries of image timestamps and their paths
    """
    rows: list[dict[str, object]] = []
    for path in jpeg_image_paths:
        pil_image = io.load_image_pil(Path(path))
        rows.append({"timestamp": get_timestamp(pil_image), "image_path": path})

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="CLI for image/polygon utilities",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,  # Python 3.7+
        metavar="COMMAND",
    )

    p1 = subparsers.add_parser(
        "polygons_to_image",
        help="Use polygons from a JSON file to create a bit mask image",
    )
    p1.add_argument(
        "--json_path", type=Path, required=True, help="Path to polygons JSON"
    )
    p1.add_argument(
        "--ref_image_path",
        type=Path,
        required=True,
        help="Path to reference image for bit mask bounds",
    )
    p1.add_argument(
        "--output_path", type=Path, required=True, help="Path to output image"
    )
    p1.set_defaults(func=cmd_polygons_to_image)

    p2 = subparsers.add_parser(
        "apply_mask_and_write",
        help="Apply a mask image to an input image and write the result",
    )
    p2.add_argument("--mask_path", type=Path, required=True, help="Path to mask image")
    p2.add_argument(
        "--image_path", type=Path, required=True, help="Path to input image"
    )
    p2.add_argument(
        "--output_path", type=Path, required=True, help="Path to output image"
    )
    p2.set_defaults(func=cmd_apply_mask_and_write)

    p3 = subparsers.add_parser(
        "image_dir_to_timeseries",
        help="Convert an image directory into a timeseries CSV (suitable for input to process)",
    )
    p3.add_argument("--image_dir", type=Path, required=True, help="Directory of images")
    p3.add_argument(
        "--output_path", type=Path, required=True, help="Path to output CSV"
    )
    p3.set_defaults(func=cmd_image_dir_to_timeseries)

    return parser


# --- Command handlers ---
def cmd_polygons_to_image(args: argparse.Namespace) -> None:
    polygons = io.load_polygons_from_json(args.json_path)
    image = io.load_image(args.ref_image_path)

    height, _width = image.shape[:2]
    # TODO: Add arg switch to allow user to choose whether polygons need to be converted (or separate the function altogether)
    polygons_t = util.convert_spatial_to_image_coords(polygons, height)
    out_image = util.polygons_to_image(polygons_t, image)
    io.write_image(out_image, args.output_path)


def cmd_apply_mask_and_write(args: argparse.Namespace) -> None:
    mask_image = io.load_image(args.mask_path)
    image = io.load_image(args.image_path)
    io.write_image(util.apply_mask(image, mask_image), args.output_path)


def cmd_image_dir_to_timeseries(args: argparse.Namespace) -> None:
    jpeg_image_paths = sorted(args.image_dir.glob("*.jpg"))
    timeseries = images_to_timeseries(jpeg_image_paths)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    timeseries.to_csv(args.output_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Dispatch
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
