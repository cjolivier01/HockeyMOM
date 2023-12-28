import os
import yaml


def get_game_config(game_id: str, root_dir: str):
    yaml_file_path = os.path.join(root_dir, "config", "games", game_id + ".yaml")
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                return yaml_content
            except yaml.YAMLError as exc:
                print(exc)
    return {}


def get_rink_config(rink: str, root_dir: str):
    yaml_file_path = os.path.join(root_dir, "config", "rinks", rink + ".yaml")
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                return yaml_content
            except yaml.YAMLError as exc:
                print(exc)
    return {}


def get_clip_box(game_id: str, root_dir: str):
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if game_config:
        game = game_config["game"]
        if "clip_box" in game:
            return game["clip_box"]
    return None
