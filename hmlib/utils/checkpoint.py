from collections import OrderedDict
from typing import Any, Dict, Union

import torch
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model, get_state_dict
from typeguard import typechecked

# Final tokens where we don't substitute
# _FINAL_LAYER_TOKENS: Set[str] = {"bn", "conv"}

DEFAULT_CHECKPOINT_REPLACEMENTS: OrderedDict[str, str] = OrderedDict(
    {
        # "_": ".",
        # "detector.backbone.": "save_detector_backbone.",
        "backbone.": "detector.backbone.",
        # "save_detector_backbone": "backbone",
    }
)


@typechecked
def _get_state_dict(
    checkpoint: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(checkpoint, str):
        checkpoint = _load_checkpoint(checkpoint, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return state_dict


@typechecked
def convert_dict(
    state_dict: Dict[str, Any],
    replacements: OrderedDict[str, str],
) -> OrderedDict[str, Any]:
    new_dict: OrderedDict[str, Any] = OrderedDict()
    for name in sorted(state_dict.keys()):
        value = state_dict[name]
        for repl_from, repl_to in replacements.items():
            name = name.replace(repl_from, repl_to)
        new_dict[name] = value
    # Make sure we didnbt overwite an actual value with the converted value
    assert len(new_dict) == len(state_dict)
    return new_dict


@typechecked
def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint: Union[str, Dict[str, Any]],
    replacements: OrderedDict[str, str] = DEFAULT_CHECKPOINT_REPLACEMENTS,
):
    device = next(iter(model.parameters())).device
    base_checkpoint = get_state_dict(model)
    base_state_dict = _get_state_dict(base_checkpoint)
    checkpoint_state_dict = _get_state_dict(_load_checkpoint(checkpoint, map_location="cpu"))

    # Convert first rather than doing, simply so that we can inspect if we'd like to
    converted_base_dict = base_state_dict
    # converted_base_dict: Dict[str, Any] = convert_dict(
    #     base_state_dict, replacements=replacements
    # )
    converted_state_dict: Dict[str, Any] = convert_dict(
        checkpoint_state_dict, replacements=replacements
    )

    new_state_dict = OrderedDict()
    found_count = 0
    not_found_count = 0
    for name, value in converted_state_dict.items():
        if name in converted_base_dict:
            # print(f"found: {name}")
            new_state_dict[name] = value
            found_count += 1
        else:
            not_found_count += 1
    base_checkpoint.update(new_state_dict)
    if found_count:
        _load_checkpoint_to_model(model, base_checkpoint)
        model.to(device)
