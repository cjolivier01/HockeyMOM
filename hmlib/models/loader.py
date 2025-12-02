from typing import Dict

from hmlib.config import get_config, get_nested_value


def get_model_config(game_id: str, model_name: str) -> Dict[str, str]:
    game_config = get_config(game_id=game_id)
    config_file = get_nested_value(game_config, f"model.{model_name}.config")
    if not config_file:
        return None, None
    checkpoint = get_nested_value(game_config, f"model.{model_name}.checkpoint")
    return config_file, checkpoint


# Legacy detector loader removed; Aspen ModelFactoryPlugin builds models from YAML
