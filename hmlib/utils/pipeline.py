"""Helpers for inspecting and mutating mmengine-style data pipelines.

These utilities operate on lists or dicts of pipeline definitions as used by
MMDetection / MMPose configs.

@see @ref hmlib.builder.DATASETS "DATASETS registry" for dataset registration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


def get_pipeline_list(data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a pipeline config into a list of dictionaries.

    @param data_pipeline: Either a list of pipeline dicts or a dict with ``\"pipeline\"`` key.
    @return: Underlying pipeline list, or ``None`` if not present.
    """
    if isinstance(data_pipeline, list):
        return data_pipeline
    else:
        return data_pipeline.get("pipeline", None)


def get_pipeline_items(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]], class_name: str
) -> List[Tuple[int, Dict[str, Any]]]:
    """Return all pipeline entries whose ``type`` matches a given class name.

    @param data_pipeline: Pipeline list or config dict.
    @param class_name: Value of the ``\"type\"`` field to search for.
    @return: List of ``(index, item_dict)`` pairs for each match.
    """
    results: List[Tuple[int, Dict[str, Any]]] = []
    pipeline_list = get_pipeline_list(data_pipeline)
    if not pipeline_list:
        return results
    for index, pipeline_item in enumerate(pipeline_list):
        if pipeline_item["type"] == class_name:
            results.append((index, pipeline_item))
    return results


def get_pipeline_item(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]], class_name: str
) -> List[Tuple[int, Dict[str, Any]]]:
    """Return a single pipeline entry by ``type``.

    @param data_pipeline: Pipeline list or config dict.
    @param class_name: Value of the ``\"type\"`` field to search for.
    @return: Matching pipeline dict or ``None`` if not found.
    @see @ref get_pipeline_items "get_pipeline_items" for multi-match queries.
    """
    results = get_pipeline_items(data_pipeline=data_pipeline, class_name=class_name)
    if not results:
        return None
    assert len(results) == 1
    return results[0][1]


def update_pipeline_item(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]], class_name: str, data=Dict[str, Any]
) -> bool:
    """Update a pipeline entry in-place if it exists.

    @param data_pipeline: Pipeline list or config dict.
    @param class_name: ``\"type\"`` value to match.
    @param data: Mapping of keys to update on the pipeline item.
    @return: ``True`` if a matching item was found and updated.
    """
    pipeline_item = get_pipeline_item(data_pipeline, class_name)
    if pipeline_item is not None:
        pipeline_item.update(data)
        return True
    return False


def get_model_item(
    model_config: Dict[str, Any], attribute: str, class_name: Optional[str] = None
) -> List[Tuple[int, Dict[str, Any]]]:
    """Placeholder for querying model configs for a given attribute.

    This helper is currently not implemented.
    """
    # results = get_pipeline_items(data_pipeline=data_pipeline, class_name=class_name)
    pass


def set_pipeline_item_attribute(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]],
    class_name: str,
    attribute: str,
    value: Any,
) -> bool:
    """Set an attribute on a pipeline item object.

    @param data_pipeline: Pipeline list or config dict.
    @param class_name: ``\"type\"`` value to match.
    @param attribute: Attribute name to set on the pipeline object.
    @param value: New attribute value.
    @return: ``True`` if the attribute was set on exactly one item.
    """
    pipeline_items = get_pipeline_items(data_pipeline=data_pipeline, class_name=class_name)
    if not pipeline_items:
        return False
    assert len(pipeline_items) == 1
    assert hasattr(pipeline_items[0], attribute)
    setattr(pipeline_items[0], attribute, value)
    return True


def replace_pipeline_class(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]],
    from_class: str,
    to_class: str,
) -> None:
    """Replace or remove pipeline entries whose ``type`` matches ``from_class``.

    @param data_pipeline: Pipeline list or config dict.
    @param from_class: Class name to search for in the ``\"type\"`` field.
    @param to_class: Replacement class name; if empty, matching entries are removed.
    """
    pipeline_list = get_pipeline_list(data_pipeline)
    if not pipeline_list:
        return
    del_count = 0
    for index, pipeline_item in enumerate(pipeline_list.copy()):
        assert isinstance(pipeline_item, dict)
        if pipeline_item["type"] == from_class:
            if to_class:
                pipeline_list[index - del_count]["type"] = to_class
            else:
                # delete it from the list
                del pipeline_list[index - del_count]
                # index into the original list is
                # shorter by one more now
                del_count += 1
