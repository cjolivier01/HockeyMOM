import argparse
import os
from functools import lru_cache
from typing import Dict, List, Optional

import yaml


def adjusted_config_path(path: str, team: str, season: str, args: argparse.Namespace):
    if team:
        path = os.path.join(path, args.team)
    if season:
        path = os.path.join(path, args.season)
    return path


def load_config_file(
    root_dir: str, config_type: str, config_name: str, merge_into_config: dict = None
):
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                if merge_into_config:
                    yaml_content = recursive_update(merge_into_config, yaml_content)
                return yaml_content
            except yaml.YAMLError as exc:
                print(exc)
    return {} if not merge_into_config else merge_into_config


def save_config_file(root_dir: str, config_type: str, config_name: str, data: dict):
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def baseline_config(root_dir: str):
    return load_config_file(root_dir=root_dir, config_type=".", config_name="baseline")


def get_game_config(game_id: str, root_dir: str):
    return load_config_file(root_dir=root_dir, config_type="games", config_name=game_id)


def save_game_config(game_id: str, root_dir: str, data: dict):
    return save_config_file(root_dir=root_dir, config_type="games", config_name=game_id, data=data)


def get_rink_config(rink: str, root_dir: str):
    return load_config_file(root_dir=root_dir, config_type="rinks", config_name=rink)


def get_camera_config(camera: str, root_dir: str):
    return load_config_file(root_dir=root_dir, config_type="camera", config_name=camera.lower())


def get_item(key: str, maps: List[Dict]):
    for map in maps:
        if map is not None and key in map:
            return map[key]
    return None


def get_config(
    root_dir: str, game_id: str, rink: Optional[str] = None, camera: Optional[str] = None
):
    """
    Get a consolidated configuration.
    Direct parameters override parameters whihc are in the higher-level yaml
    (i.e. a specified 'rink' to this function overrides the 'rink' in the game config)
    """
    consolidated_config = baseline_config(root_dir=root_dir)
    game_config = dict()
    rink_config = dict()
    camera_config = dict()
    if camera is not None:
        camera_config = get_camera_config(camera=camera, root_dir=root_dir)
    if rink is not None:
        rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    if game_id is not None:
        game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if camera is None:
        camera = get_item("camera", [game_config, rink_config])
        if camera:
            camera_config = get_camera_config(camera=camera["name"], root_dir=root_dir)
    if rink is None:
        rink = get_nested_value(game_config, "game.rink")
        if rink:
            rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    consolidated_config = recursive_update(consolidated_config, camera_config)
    consolidated_config = recursive_update(consolidated_config, rink_config)
    consolidated_config = recursive_update(consolidated_config, game_config)
    return consolidated_config


def update_config(root_dir: str, baseline_config: dict, config_type: str, config_name: str):
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    if not os.path.exists(yaml_file_path):
        return baseline_config
    config = load_config_file(root_dir=root_dir, config_type=config_type, config_name=config_name)
    return recursive_update(baseline_config, config)


@lru_cache
def get_clip_box(game_id: str, root_dir: str):
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if game_config:
        game = game_config.get("game", None)
        if game and "clip_box" in game:
            return game["clip_box"]
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
