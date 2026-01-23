import torch

from hmlib.aspen.plugins.overlay_plugin import OverlayPlugin


def should_overlay_plugin_pass_through_image_when_no_aux_data():
    plugin = OverlayPlugin(
        enabled=True,
        plot_pose=False,
        plot_jersey_numbers=False,
        plot_tracking=False,
        plot_all_detections=None,
        plot_trajectories=False,
        print_available=False,
    )
    img = torch.zeros((2, 16, 16, 3), dtype=torch.uint8)
    out = plugin({"img": img, "data": {}})
    assert out == {}
