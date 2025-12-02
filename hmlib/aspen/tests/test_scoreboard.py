import torch

from hmlib.scoreboard.scoreboard import Scoreboard


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

