from typing import Any, Tuple, Union, Dict, List


def get_pipeline_list(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if isinstance(data_pipeline, list):
        return data_pipeline
    else:
        return data_pipeline.get("pipeline", None)


def get_pipeline_items(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]], class_name: str
) -> List[Tuple[int, Dict[str, Any]]]:
    results: List[Tuple[int, Dict[str, Any]]] = []
    pipeline_list = get_pipeline_list(data_pipeline)
    if not pipeline_list:
        return results
    for index, pipeline_item in enumerate(pipeline_list):
        if pipeline_item["type"] == class_name:
            results.append((index, pipeline_item))
    return results


def set_pipeline_item_attribute(
    data_pipeline: Union[List[Dict[str, Any]], Dict[str, Any]],
    class_name: str,
    attribute: str,
    value: Any,
) -> bool:
    pipeline_items = get_pipeline_items(
        data_pipeline=data_pipeline, class_name=class_name
    )
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
