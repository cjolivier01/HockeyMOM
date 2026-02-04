import pytest


torch = pytest.importorskip("torch")

from hmlib.stitching.blender2 import BlendImageInfo, PtImageBlender


def _make_checkerboard(height: int, width: int, v0: int, v1: int) -> torch.Tensor:
    grid = (torch.arange(height).unsqueeze(1) + torch.arange(width)) % 2
    img = torch.where(grid == 0, torch.full_like(grid, v0), torch.full_like(grid, v1))
    img = img.to(torch.uint8)
    return img.unsqueeze(0).repeat(3, 1, 1)


def _make_zigzag(height: int, width: int, vmin: int, vmax: int) -> torch.Tensor:
    base = torch.arange(width, dtype=torch.float32)
    if width > 1:
        base = base / float(width - 1)
    base = base * (vmax - vmin) + vmin
    rows = []
    for row in range(height):
        row_vals = base if row % 2 == 0 else torch.flip(base, dims=[0])
        rows.append(row_vals)
    img = torch.stack(rows, dim=0).round().to(torch.uint8)
    return img.unsqueeze(0).repeat(3, 1, 1)


def _assert_pixel(output: torch.Tensor, row: int, col: int, expected: torch.Tensor) -> None:
    pix = output[0, :, row, col]
    expected = expected.to(pix.dtype)
    if torch.is_floating_point(pix):
        assert torch.allclose(pix, expected, atol=1e-3)
    else:
        assert torch.equal(pix, expected)


@pytest.mark.parametrize("laplacian_blend", [False, True])
def should_respect_alpha_masks_in_python_blender(laplacian_blend: bool) -> None:
    height, width = 6, 6
    left_img = _make_checkerboard(height, width, v0=10, v1=90)
    right_img = _make_zigzag(height, width, vmin=150, vmax=240)

    seam = torch.zeros((height, width), dtype=torch.uint8)
    seam[:, width // 2 :] = 1

    alpha_left = torch.full((height, width), 255, dtype=torch.uint8)
    alpha_right = torch.full((height, width), 255, dtype=torch.uint8)
    alpha_left[2:4, 1:3] = 0
    alpha_right[0:2, 4:6] = 0
    alpha_left[5, 5] = 0
    alpha_right[5, 5] = 0

    alpha_mask_left = alpha_left == 0
    alpha_mask_right = alpha_right == 0

    img_dtype = torch.float32 if laplacian_blend else torch.uint8
    left = left_img.to(img_dtype).unsqueeze(0)
    right = right_img.to(img_dtype).unsqueeze(0)

    blender = PtImageBlender(
        images_info=[
            BlendImageInfo(remapped_width=0, remapped_height=0, xpos=0, ypos=0),
            BlendImageInfo(remapped_width=0, remapped_height=0, xpos=0, ypos=0),
        ],
        seam_mask=seam,
        xor_mask=seam,
        dtype=img_dtype if laplacian_blend else torch.uint8,
        laplacian_blend=laplacian_blend,
        max_levels=2,
        add_alpha_channel=False,
    )

    output = blender.forward(left, alpha_mask_left, right, alpha_mask_right)

    only_right_row, only_right_col = 2, 1
    expected_right = right[0, :, only_right_row, only_right_col]
    _assert_pixel(output, only_right_row, only_right_col, expected_right)

    only_left_row, only_left_col = 0, 4
    expected_left = left[0, :, only_left_row, only_left_col]
    _assert_pixel(output, only_left_row, only_left_col, expected_left)

    neither_row, neither_col = 5, 5
    expected_zero = torch.zeros(3, dtype=output.dtype, device=output.device)
    _assert_pixel(output, neither_row, neither_col, expected_zero)
