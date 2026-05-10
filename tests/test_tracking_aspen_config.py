from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1] / "hmlib"
ASPEN_CONFIG_DIR = ROOT_DIR / "config" / "aspen"
TRACKING_CONFIGS = sorted(ASPEN_CONFIG_DIR.glob("tracking*.yaml"))


def _load_aspen_config(name: str):
    config_path = ASPEN_CONFIG_DIR / name
    cfg = yaml.safe_load(config_path.read_text())
    return cfg["aspen"]["plugins"]


def _has_path(plugins: dict, source: str, target: str) -> bool:
    seen = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if node in seen:
            continue
        seen.add(node)
        deps = plugins.get(node, {}).get("depends", []) or []
        stack.extend(dep for dep in deps if dep not in seen)
    return False


def should_save_detections_before_mutating_detector_consumers():
    for config_path in TRACKING_CONFIGS:
        plugins = _load_aspen_config(config_path.name)
        if "save_detections" not in plugins or "ice_boundaries" not in plugins:
            continue

        assert _has_path(plugins, "save_detections", "ice_boundaries"), config_path.name


def should_pass_rotation_grid_cache_setting_to_tracking_and_stitching_graphs():
    names = [path.name for path in TRACKING_CONFIGS] + ["stitching.yaml"]
    for name in names:
        plugins = _load_aspen_config(name)
        params = plugins["stitching"]["params"]

        assert params["cache_rotation_grid"] == "GLOBAL.stitching.cache_rotation_grid"
