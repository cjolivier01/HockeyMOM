import torch

import hmlib.hm_transforms  # noqa: F401 - ensure TRANSFORMS are registered
from mmcv.transforms import Compose


def _compose_color_adjust(**kwargs):
    pipeline = [
        dict(type="HmImageColorAdjust", **kwargs),
    ]
    return Compose(pipeline)


def should_white_balance_scale_rgb_chw():
    # Create a small CHW image tensor with known values
    img = torch.tensor(
        [
            [[10.0, 10.0], [10.0, 10.0]],  # R
            [[20.0, 20.0], [20.0, 20.0]],  # G
            [[30.0, 30.0], [30.0, 30.0]],  # B
        ],
        dtype=torch.float32,
    )
    data = {"img": img.clone()}

    # Apply white balance gains: R*2.0, G*1.0, B*0.5
    pipeline = _compose_color_adjust(white_balance=[2.0, 1.0, 0.5])
    out = pipeline(data)
    out_img = out["img"]

    expected = torch.tensor(
        [
            [[20.0, 20.0], [20.0, 20.0]],
            [[20.0, 20.0], [20.0, 20.0]],
            [[15.0, 15.0], [15.0, 15.0]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(out_img, expected)


def should_noop_when_no_adjustments():
    img = torch.rand(3, 4, 5, dtype=torch.float32) * 255.0
    data = {"img": img.clone()}
    pipeline = _compose_color_adjust()
    out = pipeline(data)
    assert torch.allclose(out["img"], img)


def should_apply_brightness_multiplier():
    img = torch.ones(3, 3, 3, dtype=torch.float32) * 10.0
    data = {"img": img.clone()}
    pipeline = _compose_color_adjust(brightness=2.0)
    out = pipeline(data)
    expected = torch.ones_like(img) * 20.0
    assert torch.allclose(out["img"], expected)


def should_apply_exposure_ev_positive():
    img = torch.ones(3, 3, 3, dtype=torch.float32) * 50.0
    data = {"img": img.clone()}
    pipeline = _compose_color_adjust(exposure_ev=1.0)
    out = pipeline(data)
    expected = torch.ones_like(img) * 100.0
    assert torch.allclose(out["img"], expected)


def should_apply_exposure_ev_negative():
    img = torch.ones(3, 3, 3, dtype=torch.float32) * 50.0
    data = {"img": img.clone()}
    pipeline = _compose_color_adjust(exposure_ev=-1.0)
    out = pipeline(data)
    expected = torch.ones_like(img) * 25.0
    assert torch.allclose(out["img"], expected)


def should_kelvin_white_balance_behave_reasonably():
    # Warm temperature should reduce red relative to blue (higher blue gain)
    img = torch.ones(3, 2, 2, dtype=torch.float32) * 100.0
    data = {"img": img.clone()}
    warm = _compose_color_adjust(white_balance_temp="3500k")
    out_warm = warm(data)["img"]
    # Compare channel means
    r_warm = out_warm[0].mean()
    b_warm = out_warm[2].mean()
    # For warm WB, blue should be boosted more than red
    assert b_warm > r_warm

    # Cool temperature should reduce blue relative to red (higher red gain)
    data2 = {"img": img.clone()}
    cool = _compose_color_adjust(white_balance_temp="9000k")
    out_cool = cool(data2)["img"]
    r_cool = out_cool[0].mean()
    b_cool = out_cool[2].mean()
    assert r_cool > b_cool
