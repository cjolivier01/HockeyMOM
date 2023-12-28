import os
import yaml


def get_config(root_dir: str, config_type: str, config_name: str):
    yaml_file_path = os.path.join(
        root_dir, "config", config_type, config_name + ".yaml"
    )
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                return yaml_content
            except yaml.YAMLError as exc:
                print(exc)
    return {}


def get_game_config(game_id: str, root_dir: str):
    return get_config(root_dir=root_dir, config_type="games", config_name=game_id)


def get_rink_config(rink: str, root_dir: str):
    return get_config(root_dir=root_dir, config_type="rinks", config_name=rink)


def get_camera_config(game_id: str, root_dir: str):
    return get_config(root_dir=root_dir, config_type="camera", config_name=game_id)


def get_clip_box(game_id: str, root_dir: str):
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if game_config:
        game = game_config["game"]
        if "clip_box" in game:
            return game["clip_box"]
    return None
