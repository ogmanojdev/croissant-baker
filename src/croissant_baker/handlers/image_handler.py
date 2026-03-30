"""Image file handler for datasets containing images."""

import logging
from pathlib import Path
from typing import Dict, List

from croissant_baker.handlers.base_handler import FileTypeHandler
from croissant_baker.handlers.utils import compute_file_hash

logger = logging.getLogger(__name__)

# Standard image extensions that Pillow handles natively.
_PILLOW_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".ico",
}

# Multi-band / scientific TIFF extensions that may need tifffile.
_TIFF_EXTENSIONS = {".tiff", ".tif"}

# All supported image extensions.
SUPPORTED_EXTENSIONS = _PILLOW_EXTENSIONS | _TIFF_EXTENSIONS

# MIME types for common image formats.
_MIME_TYPES: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _read_with_pillow(file_path: Path) -> Dict:
    """Read image metadata using Pillow (standard RGB/grayscale images)."""
    from PIL import Image

    with Image.open(file_path) as img:
        width, height = img.size
        # mode → number of bands: L=1, LA=2, RGB=3, RGBA=4, CMYK=4, etc.
        num_bands = len(img.getbands())
        return {
            "width": width,
            "height": height,
            "num_bands": num_bands,
            "image_format": img.format or file_path.suffix.lstrip(".").upper(),
        }


def _read_with_tifffile(file_path: Path) -> Dict:
    """Read image metadata using tifffile (multi-band / scientific TIFFs)."""
    import tifffile

    with tifffile.TiffFile(str(file_path)) as tif:
        page = tif.pages[0]
        shape = page.shape  # (height, width) or (height, width, bands)

        height = shape[0]
        width = shape[1]
        num_bands = shape[2] if len(shape) > 2 else 1

        return {
            "width": width,
            "height": height,
            "num_bands": num_bands,
            "image_format": "TIFF",
        }


def _read_image_metadata(file_path: Path) -> Dict:
    """
    Read image dimensions and band count, choosing the right backend.

    Tries Pillow first for standard formats. Falls back to tifffile for
    multi-band TIFFs that Pillow cannot decode (e.g., 12-band Sentinel-2).
    """
    suffix = file_path.suffix.lower()

    if suffix in _PILLOW_EXTENSIONS:
        return _read_with_pillow(file_path)

    # For TIFF files, try Pillow first (works for standard 1/3/4-band TIFFs),
    # then fall back to tifffile for multi-band scientific imagery.
    if suffix in _TIFF_EXTENSIONS:
        try:
            return _read_with_pillow(file_path)
        except Exception:
            return _read_with_tifffile(file_path)

    return _read_with_pillow(file_path)


class ImageHandler(FileTypeHandler):
    """
    Handler for image files (JPEG, PNG, TIFF, GIF, BMP, WebP).

    - Extracts dimensions (width, height), band count, and format
    - Uses Pillow for standard formats, tifffile for multi-band TIFFs
    - Computes SHA256 for reproducibility
    - Returns metadata with ``image_properties`` key for the builder
    """

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS

    def extract_metadata(self, file_path: Path, **kwargs) -> dict:
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            img_meta = _read_image_metadata(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read image {file_path}: {e}") from e

        suffix = file_path.suffix.lower()
        mime_type = _MIME_TYPES.get(suffix, "application/octet-stream")

        file_size = file_path.stat().st_size
        sha256 = compute_file_hash(file_path)

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_size,
            "sha256": sha256,
            "encoding_format": mime_type,
            "image_properties": {
                "width": img_meta["width"],
                "height": img_meta["height"],
                "num_bands": img_meta["num_bands"],
                "image_format": img_meta["image_format"],
            },
        }


def collect_image_summary(image_metadata_list: List[Dict]) -> Dict:
    """
    Summarize a collection of image file metadata into aggregate stats.

    Used by the metadata generator to describe an image dataset at the
    RecordSet level (e.g., total images, dimension ranges, formats).

    Args:
        image_metadata_list: List of metadata dicts from ImageHandler.

    Returns:
        Summary dict with counts, dimension ranges, and format breakdown.
    """
    if not image_metadata_list:
        return {}

    widths = []
    heights = []
    bands = []
    formats: Dict[str, int] = {}

    for meta in image_metadata_list:
        props = meta["image_properties"]
        widths.append(props["width"])
        heights.append(props["height"])
        bands.append(props["num_bands"])
        fmt = props["image_format"]
        formats[fmt] = formats.get(fmt, 0) + 1

    return {
        "num_images": len(image_metadata_list),
        "width_range": (min(widths), max(widths)),
        "height_range": (min(heights), max(heights)),
        "num_bands_range": (min(bands), max(bands)),
        "format_counts": formats,
    }
