import time
from typing import Any, Dict, Set

import pytest

from hmlib.aspen import AspenNet, Plugin


def make_graph(class_path: str, threaded: bool = False, pipeline: Dict[str, Any] | None = None):
    graph: Dict[str, Any] = {
        "plugins": {
            "test_plugin": {
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
    graph = make_graph("hmlib.aspen.plugins.test_output_keys_plugin.BadOutputPlugin")
    net = AspenNet("non_threaded", graph)

    with pytest.raises(ValueError) as excinfo:
        net({})

    msg = str(excinfo.value)
    assert "test_plugin" in msg
    assert "extra_key" in msg


def should_output_keys_be_checked_only_first_call_by_default():
    graph = make_graph("hmlib.aspen.plugins.test_output_keys_plugin.FlakyOutputPlugin")
    net = AspenNet("default_check_once", graph)

    context: Dict[str, Any] = {}
    net(context)

    # On the second call the plugin returns an undeclared key, but by default
    # AspenNet only validates output_keys() on the first call per trunk.
    context2: Dict[str, Any] = {}
    net(context2)


def should_output_keys_be_checked_every_call_when_enabled():
    graph = make_graph(
        "hmlib.aspen.plugins.test_output_keys_plugin.FlakyOutputPlugin",
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
        "hmlib.aspen.plugins.test_output_keys_plugin.BadOutputPlugin",
        threaded=True,
        pipeline={"threaded": True, "queue_size": 1},
    )
    net = AspenNet("threaded", graph)

    # First call enqueues work for the worker threads; the violation happens
    # in the worker and is captured there.
    net({})

    # Enqueue a second context to ensure the worker processes at least one
    # item and captures the violation.
    try:
        net({})
    except ValueError:
        # Worker may have already surfaced the violation; that's acceptable.
        pass

    # Allow some time for the worker thread to run and record the error.
    for _ in range(50):
        if getattr(net, "_thread_error", None) is not None:
            break
        time.sleep(0.02)

    # Once a worker has captured the exception, subsequent calls to forward()
    # should surface it via _maybe_reraise_thread_error.
    assert getattr(net, "_thread_error", None) is not None, (
        "Expected _thread_error to be set in threaded AspenNet but it was still None"
    )
    with pytest.raises(ValueError):
        net({})
