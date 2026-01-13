import torch

from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.transforms.perspective_rotation import HmPerspectiveRotation
from hmlib.transforms.scoreboard_transforms import HmCaptureScoreboard, HmRenderScoreboard


def should_scoreboard_output_fixed_shape_for_nchw_and_nhwc():
    src_pts = torch.tensor(
        [[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]], dtype=torch.float32
    )

    dest_height = 32
    dest_width = 64
    scoreboard = Scoreboard(
        src_pts=src_pts,
        dest_height=dest_height,
        dest_width=dest_width,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    img_nchw = torch.rand(1, 3, 128, 256, dtype=torch.float32)
    img_nhwc = img_nchw.permute(0, 2, 3, 1).contiguous()

    out_nchw = scoreboard(img_nchw)
    out_nhwc = scoreboard(img_nhwc)

    assert out_nchw.shape == (1, 3, dest_height, dest_width)
    assert out_nhwc.shape == (1, 3, dest_height, dest_width)

    out_nchw_2 = scoreboard(img_nchw)
    assert out_nchw_2.shape == out_nchw.shape

    assert hasattr(scoreboard, "_grid")
    assert scoreboard._grid.shape[0] == 1
    assert scoreboard._grid.shape[-1] == 2


def should_capture_and_render_scoreboard_with_fixed_shape():
    src_pts = [[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]]
    dest_height = 32
    dest_width = 64

    capture = HmCaptureScoreboard()
    render = HmRenderScoreboard(image_labels=["img"])

    img = torch.rand(1, 120, 200, 3, dtype=torch.float32)
    results = {
        "img": img.clone(),
        "scoreboard_cfg": {
            "scoreboard_points": src_pts,
            "dest_width": dest_width,
            "dest_height": dest_height,
        },
    }

    results = capture(results)
    scoreboard_img = results["scoreboard_img"]

    assert scoreboard_img.shape == (1, dest_height, dest_width, 3)

    rendered_results = render(results)
    rendered_img = rendered_results["img"]

    assert rendered_img.shape == img.shape
    assert torch.allclose(rendered_img[:, :dest_height, :dest_width, :], scoreboard_img)

    img2 = torch.rand(1, 120, 200, 3, dtype=torch.float32)
    results2 = {
        "img": img2.clone(),
        "scoreboard_cfg": {"scoreboard_points": src_pts},
    }

    results2 = capture(results2)
    scoreboard_img2 = results2["scoreboard_img"]

    assert scoreboard_img2.shape == scoreboard_img.shape

    rendered_results2 = render(results2)
    rendered_img2 = rendered_results2["img"]

    assert rendered_img2.shape == img2.shape
    assert torch.allclose(rendered_img2[:, :dest_height, :dest_width, :], scoreboard_img2)


def should_perspective_rotation_pre_clip_keep_fixed_width():
    src_pts = [[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]]

    img = torch.rand(2, 180, 320, 3, dtype=torch.float32)
    camera_boxes = torch.tensor(
        [
            [40.0, 50.0, 140.0, 110.0],
            [60.0, 40.0, 190.0, 100.0],
        ],
        dtype=torch.float32,
    )

    transform = HmPerspectiveRotation(
        fixed_edge_rotation=True,
        fixed_edge_rotation_angle=10.0,
        pre_clip=True,
        image_label="img",
        bbox_label="camera_box",
    )

    results = {
        "img": img.clone(),
        "camera_box": camera_boxes.clone(),
        "scoreboard_cfg": {"scoreboard_points": src_pts},
    }

    results = transform(results)
    rotated_imgs = results["img"]

    assert isinstance(rotated_imgs, torch.Tensor)
    assert rotated_imgs.shape[0] == 2
    w0 = rotated_imgs[0].shape[1]
    w1 = rotated_imgs[1].shape[1]
    h0 = rotated_imgs[0].shape[0]
    h1 = rotated_imgs[1].shape[0]

    assert w0 == w1
    assert h0 == h1
