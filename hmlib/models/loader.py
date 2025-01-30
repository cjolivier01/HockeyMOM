from typing import Dict

import torch
from mmdet.apis import init_detector
from mmdet.models.detectors.base import BaseDetector

from hmlib.config import get_config, get_nested_value


def get_model_config(game_id: str, model_name: str) -> Dict[str, str]:
    game_config = get_config(game_id=game_id)
    config_file = get_nested_value(game_config, f"model.{model_name}.config")
    if not config_file:
        return None, None
    checkpoint = get_nested_value(game_config, f"model.{model_name}.checkpoint")
    return config_file, checkpoint


def load_detector_model(
    game_id: str,
    model_name: str,
    device: torch.device,
    config_file: str = None,
    checkpoint: str = None,
) -> BaseDetector:
    if not config_file:
        config_file, loaded_checkpoint = get_model_config(game_id=game_id, model_name=model_name)
        if not checkpoint:
            checkpoint = loaded_checkpoint
    if config_file:
        return init_detector(config_file, checkpoint, device=device)
    return None
