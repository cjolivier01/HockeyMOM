import struct
import zlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from hmlib.stitching.seam import (
    load_canvas_seam_mask,
    read_mapping_canvas_size,
    read_png_layout,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data))
    )


def _write_grayscale_png(
    path: Path,
    pixels: list[list[int]],
    offset: tuple[int, int] | None = None,
    offset_after_image_data: bool = False,
) -> None:
    height = len(pixels)
    width = len(pixels[0])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    image_data = zlib.compress(b"".join(b"\x00" + bytes(row) for row in pixels))
    chunks = [_png_chunk(b"IHDR", ihdr)]
    offset_chunk = None
    if offset is not None:
        offset_chunk = _png_chunk(b"oFFs", struct.pack(">iiB", *offset, 0))
        if not offset_after_image_data:
            chunks.append(offset_chunk)
    chunks.append(_png_chunk(b"IDAT", image_data))
    if offset_chunk is not None and offset_after_image_data:
        chunks.append(offset_chunk)
    chunks.append(_png_chunk(b"IEND", b""))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def should_place_cropped_seam_at_png_offset(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=(1, 2))

    seam = load_canvas_seam_mask(seam_file, canvas_width=5, canvas_height=5)

    assert seam.tolist() == [
        [10, 10, 20, 20, 20],
        [10, 10, 20, 20, 20],
        [10, 10, 20, 20, 20],
        [30, 30, 40, 40, 40],
        [30, 30, 40, 40, 40],
    ]


def should_treat_a_missing_png_offset_as_the_canvas_origin(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]])

    seam = load_canvas_seam_mask(seam_file, canvas_width=4, canvas_height=3)

    assert seam.tolist() == [
        [10, 20, 20, 20],
        [30, 40, 40, 40],
        [30, 40, 40, 40],
    ]


@pytest.mark.parametrize("offset", [(-1, 0), (0, -1), (4, 0), (0, 4)])
def should_reject_a_seam_crop_outside_the_canvas(tmp_path: Path, offset: tuple[int, int]) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=offset)

    with pytest.raises(ValueError, match="outside its mapping canvas"):
        load_canvas_seam_mask(seam_file, canvas_width=5, canvas_height=5)


def should_reject_an_offset_after_image_data(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(
        seam_file,
        [[10, 20], [30, 40]],
        offset=(1, 1),
        offset_after_image_data=True,
    )

    with pytest.raises(ValueError, match="Invalid PNG oFFs chunk"):
        read_png_layout(seam_file)


def should_reject_a_corrupt_offset_chunk(tmp_path: Path) -> None:
    seam_file = tmp_path / "seam_file.png"
    _write_grayscale_png(seam_file, [[10, 20], [30, 40]], offset=(1, 1))
    png = bytearray(seam_file.read_bytes())
    offset_data = png.index(b"oFFs") + 4
    png[offset_data] ^= 1
    seam_file.write_bytes(png)

    with pytest.raises(ValueError, match="invalid CRC"):
        read_png_layout(seam_file)


def should_read_the_common_canvas_from_positioned_mapping_tiffs(tmp_path: Path) -> None:
    def write_mapping(
        path: Path,
        width: int,
        height: int,
        x: tuple[int, int],
        y: tuple[int, int],
    ) -> None:
        tifffile.imwrite(
            path,
            np.zeros((height, width), dtype=np.uint8),
            resolution=(1, 1),
            extratags=[
                (286, 5, 1, x, False),
                (287, 5, 1, y, False),
            ],
        )

    left = tmp_path / "mapping_0000.tif"
    right = tmp_path / "mapping_0001.tif"
    write_mapping(left, width=4, height=3, x=(15, 1), y=(26, 1))
    write_mapping(right, width=5, height=4, x=(18, 1), y=(24, 1))

    assert read_mapping_canvas_size([left, right]) == (8, 5)


def should_quantize_mapping_positions_before_normalizing_the_canvas(tmp_path: Path) -> None:
    def write_mapping(path: Path, x: tuple[int, int]) -> None:
        tifffile.imwrite(
            path,
            np.zeros((100, 200), dtype=np.uint8),
            resolution=(1, 1),
            extratags=[
                (286, 5, 1, x, False),
                (287, 5, 1, (0, 1), False),
            ],
        )

    left = tmp_path / "mapping_0000.tif"
    right = tmp_path / "mapping_0001.tif"
    write_mapping(left, x=(106, 10))
    write_mapping(right, x=(604, 10))

    # Playback rounds the absolute positions to 11 and 60 before subtracting
    # the common origin, giving a 49-pixel displacement and a 249-pixel canvas.
    assert read_mapping_canvas_size([left, right]) == (249, 100)
