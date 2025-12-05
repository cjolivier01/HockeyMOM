import time
from typing import Any, Dict, Set

import pytest

from hmlib.aspen import AspenNet, Plugin


class BadOutputPlugin(Plugin):
    """Plugin that always returns an undeclared key."""

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return {"allowed_key"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        return {"allowed_key": 1, "extra_key": 2}


class FlakyOutputPlugin(Plugin):
    """Plugin that only returns an extra key on subsequent calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def input_keys(self) -> Set[str]:
        return set()

    def output_keys(self) -> Set[str]:
        return {"value"}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            return {"value": self.calls}
        return {"value": self.calls, "extra_key": 99}


def make_graph(class_path: str, threaded: bool = False, pipeline: Dict[str, Any] | None = None):
    graph: Dict[str, Any] = {
        "trunks": {
            "test_trunk": {
                "class": class_path,
                "depends": [],
                "params": {},
            }
        }
    }
    if threaded:
        graph["threaded_trunks"] = True
    if pipeline:
        graph["pipeline"] = pipeline
    return graph


def should_output_keys_violation_raise_non_threaded():
    graph = make_graph("test_aspen_output_keys.BadOutputPlugin")
    net = AspenNet("non_threaded", graph)

    with pytest.raises(ValueError) as excinfo:
        net({})

    msg = str(excinfo.value)
    assert "test_trunk" in msg
    assert "extra_key" in msg


def should_output_keys_be_checked_only_first_call_by_default():
    graph = make_graph("test_aspen_output_keys.FlakyOutputPlugin")
    net = AspenNet("default_check_once", graph)

    context: Dict[str, Any] = {}
    net(context)

    # On the second call the plugin returns an undeclared key, but by default
    # AspenNet only validates output_keys() on the first call per trunk.
    context2: Dict[str, Any] = {}
    net(context2)


def should_output_keys_be_checked_every_call_when_enabled():
    graph = make_graph(
        "test_aspen_output_keys.FlakyOutputPlugin",
        threaded=False,
        pipeline={"check_output_keys_each_time": True},
    )
    net = AspenNet("check_every_call", graph)

    context: Dict[str, Any] = {}
    net(context)

    context2: Dict[str, Any] = {}
    with pytest.raises(ValueError):
        net(context2)


def should_output_keys_violation_be_propagated_in_threaded_mode():
    graph = make_graph(
        "test_aspen_output_keys.BadOutputPlugin",
        threaded=True,
        pipeline={"threaded": True, "queue_size": 1},
    )
    net = AspenNet("threaded", graph)

    # First call enqueues work for the worker threads; the violation happens
    # in the worker and is captured there.
    net({})

    # Subsequent calls to forward() should surface the exception raised in
    # the worker thread.
    raised = False
    for _ in range(10):
        try:
            net({})
        except ValueError:
            raised = True
            break
        time.sleep(0.01)

    assert raised, "Expected ValueError from threaded AspenNet but none was raised"
