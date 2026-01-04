"""Game- and camera-specific YAML configuration helpers.

This module loads, merges and saves configuration files used throughout
HockeyMOM pipelines (games, rinks, cameras and private overrides).

@see @ref hmlib.hm_opts.hm_opts "hm_opts" for CLI flags that drive these configs.
@see @ref hmlib.game_audio.transfer_audio "transfer_audio" for one consumer.
"""

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

import hmlib
from hmlib.bbox.box_functions import scale_bbox_with_constraints
from hmlib.log import get_logger

GAME_DIR_BASE: str = os.path.join(os.environ["HOME"], "Videos")
ROOT_DIR: str = os.path.dirname(os.path.abspath(hmlib.__file__))


@dataclass
class Game:
    game_id: Optional[str] = None
    season: Optional[str] = None
    team: Optional[str] = None


def get_root_dir() -> str:
    """Return the root directory of the hmlib installation."""
    return ROOT_DIR


def prepend_root_dir(path: str) -> str:
    """Join a relative path against :data:`ROOT_DIR` if needed."""
    if not path:
        return ROOT_DIR
    if "://" in path:
        # Likely a URL
        return path
    if path[0] != "/":
        return os.path.join(ROOT_DIR, path)
    return path


def get_game_dir(game_id: str, assert_exists: bool = True) -> Optional[str]:
    """Return the video directory for a given game id.

    @param game_id: Game identifier string.
    @param assert_exists: If True, raise if the directory does not exist.
    @return: Absolute path to ``$HOME/Videos/<game_id>`` or ``None``.
    """
    if not game_id:
        raise AttributeError("No valid Game ID specified")
    game_video_dir = os.path.join(GAME_DIR_BASE, game_id)
    if os.path.isdir(game_video_dir):
        return game_video_dir
    if assert_exists:
        raise AssertionError(f"No game direectory found for game id: {game_id}")
    return None


# TODO: implement passing all of this in cleanly somehow
def adjusted_config_path(path: str, team: str, season: str, args: argparse.Namespace):
    if team:
        path = os.path.join(path, args.team)
    if season:
        path = os.path.join(path, args.season)
    return path


def get_game_config_file_name(game: Game, root_dir: Optional[str] = None) -> Path:
    # Our first try is in the game dir
    game_dir = Path(GAME_DIR_BASE) / game.game_id / "config.yaml"
    if os.path.exists(game_dir) and not os.path.isdir(game_dir):
        return game_dir
    return Path(root_dir) / "config" / "games" / game.game_id


def load_config_file_yaml(yaml_file_path: str, merge_into_config: dict = None):
    """Load a YAML config from disk and optionally merge into a base dict."""
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                if yaml_content is None:
                    # Empty file
                    return {}
                if merge_into_config:
                    yaml_content = recursive_update(merge_into_config, yaml_content)
                return yaml_content
            except yaml.YAMLError as exc:
                get_logger(__name__).exception(
                    "Failed to parse YAML config %s: %s", yaml_file_path, exc
                )
                raise
    return {} if not merge_into_config else merge_into_config


def load_yaml_files_ordered(
    paths: Sequence[str], base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load multiple YAML files in order and merge them into a dictionary.

    Later files override earlier values and add new fields.

    @param paths: Sequence of YAML file paths (absolute or ROOT_DIR-relative).
    @param base: Optional starting dictionary to merge into.
    @return: Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {} if base is None else dict(base)
    for p in paths:
        if not p:
            continue
        try:
            # Allow both absolute and ROOT_DIR-relative paths
            yaml_path = p
            if not os.path.isabs(yaml_path):
                candidate = os.path.join(ROOT_DIR, yaml_path)
                if os.path.exists(candidate):
                    yaml_path = candidate
            y = load_config_file_yaml(yaml_path)
            if y:
                merged = recursive_update(merged, y)
        except Exception:
            # Re-raise with additional context
            raise
    return merged


def load_config_file(
    config_type: str,
    config_name: str,
    merge_into_config: Optional[Dict[str, Any]] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if root_dir is None:
        root_dir = ROOT_DIR
    return load_config_file_yaml(
        os.path.join(root_dir, "config", config_type, config_name + ".yaml"),
        merge_into_config=merge_into_config,
    )


def save_config_file(root_dir: str, config_type: str, config_name: str, data: dict):
    if root_dir is None:
        root_dir = ROOT_DIR
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def baseline_config(root_dir: str) -> Dict[str, Any]:
    """Load the baseline configuration from ``config/baseline.yaml``."""
    return load_config_file(root_dir=root_dir, config_type=".", config_name="baseline")


def get_game_config_private(
    game_id: str,
    merge_into_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return load_config_file_yaml(
        yaml_file_path=os.path.join(GAME_DIR_BASE, game_id, "config.yaml"),
        merge_into_config=merge_into_config,
    )


def get_game_config(game_id: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    config_public = load_config_file(root_dir=root_dir, config_type="games", config_name=game_id)
    config_private = get_game_config_private(game_id=game_id)
    consolidated_config = recursive_update(config_public, config_private)
    return consolidated_config


def save_private_config(game_id: str, data: Dict[str, Any], verbose: bool = True):
    yaml_file_path = os.path.join(GAME_DIR_BASE, game_id, "config.yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(data, stream=file, sort_keys=False)
    if verbose:
        get_logger(__name__).info("Saved private config to %s", yaml_file_path)


def save_game_config(game_id: str, data: dict, root_dir: Optional[str] = None):
    return save_config_file(root_dir=root_dir, config_type="games", config_name=game_id, data=data)


def get_rink_config(rink: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_config_file(root_dir=root_dir, config_type="rinks", config_name=rink)


def get_camera_config(camera: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_config_file(root_dir=root_dir, config_type="camera", config_name=camera.lower())


def get_item(key: str, maps: List[Dict]):
    for map in maps:
        if map is not None and key in map:
            return map[key]
    return None


def get_config(
    root_dir: Optional[str] = None,
    game_id: Optional[str] = None,
    rink: Optional[str] = None,
    camera: Optional[str] = None,
    resolve_globals: bool = True,
):
    """Return a consolidated configuration from baseline + rink + camera + game.

    Direct parameters override higher-level YAML (e.g. an explicit ``rink``
    argument overrides the rink specified in the game config).
    """
    consolidated_config: Dict[str, Any] = baseline_config(root_dir=root_dir)
    game_config: Dict[str, Any] = dict()
    rink_config: Dict[str, Any] = dict()
    camera_config: Dict[str, Any] = dict()
    private_config: Dict[str, Any] = dict()
    if camera is not None:
        camera_config = get_camera_config(camera=camera, root_dir=root_dir)
    if rink is not None:
        rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    if game_id is not None:
        game_config = get_game_config(game_id=game_id, root_dir=root_dir)
        private_config = get_game_config_private(game_id=game_id)
    if camera is None:
        camera = get_item("camera", [game_config, rink_config])
        if isinstance(camera, str):
            camera_config = get_camera_config(camera=camera, root_dir=root_dir)
        elif camera and isinstance(camera, dict) and "name" in camera:
            camera_config = get_camera_config(camera=camera["name"], root_dir=root_dir)
    if rink is None:
        rink = get_nested_value(game_config, "game.rink")
        if rink:
            rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    consolidated_config = recursive_update(consolidated_config, camera_config)
    consolidated_config = recursive_update(consolidated_config, rink_config)
    consolidated_config = recursive_update(consolidated_config, game_config)
    consolidated_config = recursive_update(consolidated_config, private_config)
    if resolve_globals:
        consolidated_config = resolve_global_refs(consolidated_config)
    return consolidated_config


def update_config(
    baseline_config: dict, config_type: str, config_name: str, root_dir: Optional[str] = None
):
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    if not os.path.exists(yaml_file_path):
        return baseline_config
    config = load_config_file(root_dir=root_dir, config_type=config_type, config_name=config_name)
    return recursive_update(baseline_config, config)


@lru_cache
def get_clip_box(game_id: str, root_dir: Optional[str] = None, use_rink_boundary: bool = False):
    """Return the configured clip box for a game, optionally derived from rink.

    @param game_id: Game identifier.
    @param root_dir: Optional config root; defaults to :data:`ROOT_DIR`.
    @param use_rink_boundary: If True, fall back to rink boundary bbox.
    @return: Clip box as ``[x1, y1, x2, y2]`` or ``None``.
    """
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if game_config:
        game = game_config.get("game", None)
        if game and "clip_box" in game:
            return game["clip_box"]
        if use_rink_boundary:
            # Alternatively, use the rink boundary box
            rink_combined_bbox = get_nested_value(game_config, "rink.ice_contours_combined_bbox")
            if rink_combined_bbox:
                rink_scaled_bbox = scale_bbox_with_constraints(
                    bbox=rink_combined_bbox,
                    ratio_x=1.1,
                    ratio_y=1.1,
                    min_x=0,
                    min_y=0,
                    max_x=float("inf"),
                    max_y=float("inf"),
                )
                rink_scaled_bbox = [int(i) for i in rink_scaled_bbox]
                return rink_scaled_bbox
    return None


#
# Dict utilities
#
def recursive_update(original, update):
    """
    Recursively update the original dictionary with the update dictionary.
    If a key in the original dictionary is not present in the update dictionary,
    its value is preserved.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            recursive_update(original[key], value)
        else:
            original[key] = value
    return original


def get_nested_value(dct, key_str, default_value=None):
    """
    Retrieve a value from a nested dictionary using a dot-separated key string.

    Parameters:
    - dct (dict): The dictionary to search.
    - key_str (str): The dot-separated key string.

    Returns:
    - The value if found, otherwise None.
    """
    keys = key_str.split(".")
    current = dct

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value

    return current


def set_nested_value(dct, key_str, set_to, noset_value=None):
    if noset_value is None and set_to is None:
        return get_nested_value(dct, key_str, set_to)
    if set_to == noset_value:
        return get_nested_value(dct, key_str, noset_value)

    keys = key_str.split(".")
    current = dct

    for i, key in enumerate(keys):
        if isinstance(current, dict) and key in current:
            if i == len(keys) - 1:
                current[key] = set_to
            else:
                current = current[key]
        else:
            if i == len(keys) - 1:
                current[key] = set_to
            else:
                current[key] = dict()
                current = current[key]
    return get_nested_value(dct, key_str)


GLOBAL_REF_PREFIX = "GLOBAL."


def _lookup_path(config: Dict[str, Any], path_parts: Sequence[str]) -> Tuple[bool, Any]:
    cur: Any = config
    for key in path_parts:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False, None
    return True, cur


def _resolve_global_value(
    config: Dict[str, Any], value: Any, seen: Optional[Set[str]] = None
) -> Any:
    if seen is None:
        seen = set()
    if isinstance(value, str) and value.startswith(GLOBAL_REF_PREFIX):
        path_str = value[len(GLOBAL_REF_PREFIX) :]
        if path_str in seen:
            return value
        seen.add(path_str)
        ok, resolved = _lookup_path(config, [p for p in path_str.split(".") if p])
        if not ok:
            return value
        return _resolve_global_value(config, resolved, seen)
    return value


def resolve_global_refs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ``GLOBAL.*`` string references with values from the merged config.

    Example::
        brightness: GLOBAL.camera.color.brightness

    Args:
        config: Merged configuration dictionary.
    Returns:
        The same dict with references resolved in-place.
    """

    def _walk(node: Any, root: Dict[str, Any]) -> Any:
        if isinstance(node, dict):
            for k, v in node.items():
                node[k] = _walk(v, root)
            return node
        if isinstance(node, list):
            for i, v in enumerate(node):
                node[i] = _walk(v, root)
            return node
        return _resolve_global_value(root, node)

    _walk(config, config)
    return config
