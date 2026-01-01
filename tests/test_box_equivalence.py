import sys
from typing import Dict, Tuple


def _torch():
    try:
        import torch  # type: ignore
    except Exception:
        print("SKIP: torch not available", file=sys.stderr)
        sys.exit(0)
    return torch


def _core():
    try:
        import hockeymom.core as core  # type: ignore
    except Exception:
        print("SKIP: hockeymom.core not available", file=sys.stderr)
        sys.exit(0)
    return core


def make_cpp_config(core, arena_box: Tuple[float, float, float, float], cfg: Dict):
    AllLivingBoxConfig = core.AllLivingBoxConfig
    BBox = core.BBox
    c = AllLivingBoxConfig()
    # Translation constraints
    c.translation_enabled = True
    c.max_speed_x = cfg.get("max_speed_x", 20.0)
    c.max_speed_y = cfg.get("max_speed_y", 20.0)
    c.max_accel_x = cfg.get("max_accel_x", 5.0)
    c.max_accel_y = cfg.get("max_accel_y", 5.0)
    c.stop_translation_on_dir_change = cfg.get("stop_on_change", False)
    c.stop_translation_on_dir_change_delay = cfg.get("stop_delay", 0)
    c.cancel_stop_on_opposite_dir = cfg.get("cancel", False)
    c.time_to_dest_speed_limit_frames = cfg.get("time_to_dest_speed_limit_frames", 10)
    c.dynamic_acceleration_scaling = 0.0
    c.arena_angle_from_vertical = 30.0
    c.arena_box = BBox(*arena_box)
    # Disable sticky/damping features for determinism unless asked for
    c.sticky_translation = cfg.get("sticky_translation", False)
    c.sticky_size_ratio_to_frame_width = cfg.get("sticky_size_ratio_to_frame_width", 10.0)
    c.sticky_translation_gaussian_mult = cfg.get("sticky_translation_gaussian_mult", 5.0)
    c.unsticky_translation_size_ratio = cfg.get("unsticky_translation_size_ratio", 0.75)
    c.pan_smoothing_alpha = cfg.get("pan_smoothing_alpha", 0.0)
    # Resizing constraints (we keep size constant during test)
    c.resizing_enabled = True
    c.max_speed_w = cfg.get("max_speed_w", 10.0)
    c.max_speed_h = cfg.get("max_speed_h", 10.0)
    c.max_accel_w = cfg.get("max_accel_w", 3.0)
    c.max_accel_h = cfg.get("max_accel_h", 3.0)
    c.min_width = 10.0
    c.min_height = 10.0
    c.max_width = arena_box[2] - arena_box[0]
    c.max_height = arena_box[3] - arena_box[1]
    c.stop_resizing_on_dir_change = cfg.get("stop_resizing_on_dir_change", True)
    c.sticky_sizing = cfg.get("sticky_sizing", False)
    # Living config
    c.scale_dest_width = 1.0
    c.scale_dest_height = 1.0
    c.clamp_scaled_input_box = True
    return c


def make_py_box(torch, arena_box_t, init_box_t, cfg: Dict):
    from hmlib.camera.moving_box import MovingBox

    # Map cpp config into MovingBox params
    return MovingBox(
        label="py",
        bbox=init_box_t.clone(),
        arena_box=arena_box_t.clone(),
        max_speed_x=torch.tensor(cfg.get("max_speed_x", 20.0), dtype=torch.float),
        max_speed_y=torch.tensor(cfg.get("max_speed_y", 20.0), dtype=torch.float),
        max_accel_x=torch.tensor(cfg.get("max_accel_x", 5.0), dtype=torch.float),
        max_accel_y=torch.tensor(cfg.get("max_accel_y", 5.0), dtype=torch.float),
        max_width=arena_box_t[2],
        max_height=arena_box_t[3],
        stop_on_dir_change=cfg.get("stop_on_change", False),
        stop_on_dir_change_delay=int(cfg.get("stop_delay", 0)),
        cancel_stop_on_opposite_dir=bool(cfg.get("cancel", False)),
        time_to_dest_speed_limit_frames=int(cfg.get("time_to_dest_speed_limit_frames", 10)),
        pan_smoothing_alpha=float(cfg.get("pan_smoothing_alpha", 0.0)),
        sticky_translation=bool(cfg.get("sticky_translation", False)),
        sticky_sizing=bool(cfg.get("sticky_sizing", False)),
        device="cpu",
    )


def make_dest_sequence_x(torch, init_box_t, step: float, n1: int, n2: int):
    """Move right (n1 steps), then left (n2 steps), return list of TLBR tensors with same size."""
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Right
    for i in range(n1):
        cc = torch.tensor([c[0] + step * (i + 1), c[1]], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    # Left
    last = c[0] + step * n1
    for j in range(n2):
        cc = torch.tensor([last - step * (j + 1), c[1]], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    return dests


def make_dest_sequence_y(torch, init_box_t, step: float, n1: int, n2: int):
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Down (increasing y)
    for i in range(n1):
        cc = torch.tensor([c[0], c[1] + step * (i + 1)], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    # Up
    last = c[1] + step * n1
    for j in range(n2):
        cc = torch.tensor([c[0], last - step * (j + 1)], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    return dests


def _to_cpp_bbox(core, t):
    return core.BBox(float(t[0]), float(t[1]), float(t[2]), float(t[3]))


def _close(a: float, b: float, eps=1e-3):
    return abs(a - b) <= eps


def _assert_boxes_close(py_t, cpp_b, eps=1e-2):
    assert _close(float(py_t[0]), cpp_b.left, eps)
    assert _close(float(py_t[1]), cpp_b.top, eps)
    assert _close(float(py_t[2]), cpp_b.right, eps)
    assert _close(float(py_t[3]), cpp_b.bottom, eps)


def _run_case(cfg: Dict, axis: str):
    torch = _torch()
    core = _core()

    # Arena and initial box
    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    # Reasonable constraints to see behavior
    cfg_full = dict(
        max_speed_x=30.0,
        max_speed_y=30.0,
        max_accel_x=10.0,
        max_accel_y=10.0,
        stop_on_change=cfg.get("stop_on_change", False),
        stop_delay=cfg.get("stop_delay", 0),
        cancel=cfg.get("cancel", False),
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
        sticky_sizing=False,
    )

    cpp_cfg = make_cpp_config(core, arena, cfg_full)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)

    py_box = make_py_box(torch, arena_t, init_box_t, cfg_full)

    # Build destination sequence with a direction change and potential cancel
    step = 20.0
    pre = 4
    post = 6
    if axis == "x":
        dests = make_dest_sequence_x(torch, init_box_t, step, pre, post)
    else:
        dests = make_dest_sequence_y(torch, init_box_t, step, pre, post)

    # If cancel requested, after first reverse step, flip again to cause cancel
    if cfg_full["cancel"] and cfg_full["stop_delay"] > 0:
        # Flip the second reverse step back the other way
        idx = pre + 1
        if axis == "x":
            # Move to the right again
            base = dests[idx - 1]
            shift = torch.tensor([step, 0.0, step, 0.0], dtype=torch.float)
            dests[idx] = base + shift
        else:
            base = dests[idx - 1]
            shift = torch.tensor([0.0, step, 0.0, step], dtype=torch.float)
            dests[idx] = base + shift

    # Step both in lock-step and compare
    for i, d in enumerate(dests):
        cpp_new = cpp_box.forward(_to_cpp_bbox(core, d))
        py_new = py_box.forward(d)
        _assert_boxes_close(py_new, cpp_new)

        # Compare core translation state values (current speeds)
        tstate = cpp_box.translation_state()
        # Python internals
        vx = float(py_box._current_speed_x)
        vy = float(py_box._current_speed_y)
        assert _close(vx, float(tstate.current_speed_x), 5e-2)
        assert _close(vy, float(tstate.current_speed_y), 5e-2)

        # If cancel path: in the cancel step expect flags set (C++) and flashes (Python)
        if cfg_full["cancel"] and cfg_full["stop_delay"] > 0 and i == pre + 1:
            # During cancel frame, flags should be true on at least the relevant axis
            assert bool(
                getattr(tstate, "canceled_stop_x", False)
                or getattr(tstate, "canceled_stop_y", False)
            )
            assert bool(py_box._cancel_stop_x_flash or py_box._cancel_stop_y_flash)


def main():
    cases = [
        (dict(stop_delay=0, stop_on_change=False, cancel=False), "x"),
        (dict(stop_delay=0, stop_on_change=True, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=False, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=True), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=True), "y"),
    ]
    for cfg, axis in cases:
        _run_case(cfg, axis)
    # Extra: ensure braking doesn't start when not moving enough
    _should_not_brake_when_not_moving()
    print("box equivalence tests: OK")


if __name__ == "__main__":
    main()


def _should_not_brake_when_not_moving():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=60.0,
        max_speed_y=60.0,
        max_accel_x=1.0,
        max_accel_y=1.0,
        stop_on_change=True,
        stop_delay=6,
        cancel=False,
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    # Step 1: small accel to the right (x increasing)
    step = 10.0
    dest1 = init_box_t + torch.tensor([step, 0.0, step, 0.0], dtype=torch.float)
    cpp_box.forward(_to_cpp_bbox(core, dest1))
    py_box.forward(dest1)

    # Step 2: reverse direction (x decreasing), but speed < max_speed/6 so no braking should start
    dest2 = dest1 + torch.tensor([-step, 0.0, -step, 0.0], dtype=torch.float)
    cpp_box.forward(_to_cpp_bbox(core, dest2))
    py_box.forward(dest2)

    tstate = cpp_box.translation_state()
    assert int(getattr(tstate, "stop_delay_x", 0) or 0) == 0
    assert int(py_box._stop_delay_x) == 0
