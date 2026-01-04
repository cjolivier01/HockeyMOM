from __future__ import annotations

import threading

from hmlib.aspen import AspenNet


def _make_graph(plugins):
    return {
        "plugins": plugins,
        "threaded_trunks": True,
        "pipeline": {
            "threaded": True,
            "graph": True,
            "queue_size": 1,
        },
    }


def should_threaded_graph_run_sibling_nodes_concurrently():
    shared = {
        "log": {},
        "lock": threading.Lock(),
        "barrier": threading.Barrier(2),
        "barrier_timeout": 2.0,
        "done_event": threading.Event(),
        "expected": 2,
    }

    graph = _make_graph(
        {
            "root": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": [],
                "params": {"name": "root"},
            },
            "left": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.BarrierPlugin",
                "depends": ["root"],
                "params": {"name": "left", "mark_done": True},
            },
            "right": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.BarrierPlugin",
                "depends": ["root"],
                "params": {"name": "right", "mark_done": True},
            },
        }
    )

    net = AspenNet("graph_concurrent", graph, shared=shared)
    net({"seq": 0})

    assert shared["done_event"].wait(timeout=2.0), "Expected sibling nodes to run concurrently"
    net._maybe_reraise_thread_error()
    net.stop()


def should_threaded_graph_preserve_plugin_order():
    shared = {
        "log": {},
        "lock": threading.Lock(),
        "done_event": threading.Event(),
        "expected": 3,
    }

    graph = _make_graph(
        {
            "root": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": [],
                "params": {"name": "root"},
            },
            "left": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["root"],
                "params": {"name": "left", "delay": 0.01},
            },
            "right": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["root"],
                "params": {"name": "right", "delay": 0.0},
            },
            "join": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["left", "right"],
                "params": {"name": "join", "mark_done": True},
            },
        }
    )

    net = AspenNet("graph_order", graph, shared=shared)
    for seq in range(3):
        net({"seq": seq})

    assert shared["done_event"].wait(timeout=3.0), "Expected join node to process all contexts"
    net._maybe_reraise_thread_error()
    net.stop()

    for name, seqs in shared["log"].items():
        assert seqs == [0, 1, 2], f"Out-of-order execution for {name}: {seqs}"
