import cv2
import numpy as np

from hmlib.stitching.configure_stitching import _save_stitched_reference_frame


def should_save_lzw_tiff_reference_without_imagecodecs(tmp_path):
    panorama = np.zeros((8, 12, 4), dtype=np.uint8)
    panorama[..., 0] = 17
    panorama[..., 1] = 83
    panorama[..., 2] = 211
    panorama[..., 3] = 255
    panorama_path = tmp_path / "panorama.tif"
    assert cv2.imwrite(
        str(panorama_path),
        panorama,
        [cv2.IMWRITE_TIFF_COMPRESSION, 5],
    )

    _save_stitched_reference_frame(tmp_path)

    saved = cv2.imread(str(tmp_path / "s.png"), cv2.IMREAD_UNCHANGED)
    assert saved is not None
    assert saved.shape == (8, 12, 3)
    assert np.array_equal(saved, panorama[..., :3])
