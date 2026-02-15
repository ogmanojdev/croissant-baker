"""Tests for image file handler."""

from pathlib import Path

import pytest

from croissant_maker.handlers.image_handler import (
    ImageHandler,
    collect_image_summary,
)


@pytest.fixture
def handler() -> ImageHandler:
    return ImageHandler()


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("photo.jpg", True),
        ("photo.jpeg", True),
        ("photo.JPG", True),
        ("scan.png", True),
        ("scan.PNG", True),
        ("frame.gif", True),
        ("icon.bmp", True),
        ("hero.webp", True),
        ("satellite.tiff", True),
        ("satellite.tif", True),
        ("satellite.TIFF", True),
        ("data.csv", False),
        ("model.parquet", False),
        ("readme.txt", False),
        ("record.hea", False),
    ],
)
def test_can_handle(handler: ImageHandler, name: str, expected: bool) -> None:
    assert handler.can_handle(Path(name)) == expected


# ---------------------------------------------------------------------------
# extract_metadata — standard JPG images (glaucoma fundus)
# ---------------------------------------------------------------------------


@pytest.fixture
def glaucoma_image_path() -> Path:
    """Path to a sample JPG from the glaucoma fundus dataset."""
    p = (
        Path(__file__).parent
        / "data"
        / "input"
        / "glaucoma_fundus"
        / "Images"
        / "0_0.jpg"
    )
    if not p.exists():
        pytest.skip(f"Glaucoma fundus image not found at {p}")
    return p


def test_extract_metadata_jpg(handler: ImageHandler, glaucoma_image_path: Path) -> None:
    meta = handler.extract_metadata(glaucoma_image_path)

    assert meta["file_name"] == "0_0.jpg"
    assert meta["encoding_format"] == "image/jpeg"
    assert meta["file_size"] > 0
    assert len(meta["sha256"]) == 64

    props = meta["image_properties"]
    assert props["width"] > 0
    assert props["height"] > 0
    assert props["num_bands"] in (1, 3, 4)
    assert props["image_format"] == "JPEG"


# ---------------------------------------------------------------------------
# extract_metadata — multi-band TIFF images (satellite)
# ---------------------------------------------------------------------------


@pytest.fixture
def satellite_tiff_path() -> Path:
    """Path to a sample TIFF from the satellite dataset."""
    p = (
        Path(__file__).parent
        / "data"
        / "input"
        / "satellite_public_health"
        / "images"
        / "5001"
        / "image_2016-01-03.tiff"
    )
    if not p.exists():
        pytest.skip(f"Satellite TIFF not found at {p}")
    return p


def test_extract_metadata_tiff(
    handler: ImageHandler, satellite_tiff_path: Path
) -> None:
    meta = handler.extract_metadata(satellite_tiff_path)

    assert meta["file_name"] == "image_2016-01-03.tiff"
    assert meta["encoding_format"] == "image/tiff"
    assert meta["file_size"] > 0
    assert len(meta["sha256"]) == 64

    props = meta["image_properties"]
    assert props["width"] > 0
    assert props["height"] > 0
    # Sentinel-2 images have 12 bands
    assert props["num_bands"] == 12
    assert props["image_format"] == "TIFF"


# ---------------------------------------------------------------------------
# extract_metadata — error cases
# ---------------------------------------------------------------------------


def test_extract_metadata_not_found(handler: ImageHandler) -> None:
    with pytest.raises(FileNotFoundError):
        handler.extract_metadata(Path("/nonexistent/image.jpg"))


def test_extract_metadata_corrupt_file(handler: ImageHandler, tmp_path: Path) -> None:
    bad_img = tmp_path / "corrupt.jpg"
    bad_img.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Failed to read image"):
        handler.extract_metadata(bad_img)


# ---------------------------------------------------------------------------
# collect_image_summary
# ---------------------------------------------------------------------------


def test_collect_image_summary_empty() -> None:
    assert collect_image_summary([]) == {}


def test_collect_image_summary() -> None:
    metas = [
        {
            "image_properties": {
                "width": 100,
                "height": 200,
                "num_bands": 3,
                "image_format": "JPEG",
            }
        },
        {
            "image_properties": {
                "width": 640,
                "height": 480,
                "num_bands": 3,
                "image_format": "JPEG",
            }
        },
        {
            "image_properties": {
                "width": 256,
                "height": 256,
                "num_bands": 12,
                "image_format": "TIFF",
            }
        },
    ]
    summary = collect_image_summary(metas)

    assert summary["num_images"] == 3
    assert summary["width_range"] == (100, 640)
    assert summary["height_range"] == (200, 480)
    assert summary["num_bands_range"] == (3, 12)
    assert summary["format_counts"] == {"JPEG": 2, "TIFF": 1}


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------


def test_image_handler_registered() -> None:
    """ImageHandler should be discoverable via the global registry."""
    from croissant_maker.handlers.registry import find_handler, register_all_handlers

    register_all_handlers()
    assert find_handler(Path("photo.jpg")) is not None
    assert find_handler(Path("scan.png")) is not None
    assert find_handler(Path("satellite.tiff")) is not None
