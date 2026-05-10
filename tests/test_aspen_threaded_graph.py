from __future__ import annotations

import threading

import pytest
import torch

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
                "class": "hmlib.aspen.plugins.join_plugin.JoinPlugin",
                "depends": ["left", "right"],
                "params": {
                    "required_plugins": ["left", "right"],
                    "output_key": "joined",
                },
            },
            "sink": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["join"],
                "params": {"name": "sink", "mark_done": True},
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


def should_threaded_graph_stop_drain_inflight_contexts():
    shared = {
        "log": {},
        "lock": threading.Lock(),
    }

    graph = _make_graph(
        {
            "root": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": [],
                "params": {"name": "root", "delay": 0.02},
            },
            "sink": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["root"],
                "params": {"name": "sink", "delay": 0.02},
            },
        }
    )
    graph["pipeline"]["max_concurrent"] = 3
    graph["pipeline"]["queue_size"] = 3

    net = AspenNet("graph_stop_drain", graph, shared=shared)
    for seq in range(3):
        net({"seq": seq})

    net.stop(wait=True)

    assert shared["log"].get("root") == [0, 1, 2]
    assert shared["log"].get("sink") == [0, 1, 2]


class _FakeThread:
    name = "stuck-worker"

    def __init__(self):
        self.join_timeouts = []

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


class _FakeQueue:
    def __init__(self):
        self.closed = False
        self.put_calls = []

    def put(self, item, block=True):
        self.put_calls.append((item, block))

    def close(self):
        self.closed = True


def should_threaded_graph_stop_surface_drain_timeout_without_unbounded_join():
    thread = _FakeThread()
    graph_queue = _FakeQueue()
    net = object.__new__(AspenNet)
    torch.nn.Module.__init__(net)
    net.threaded_trunks = True
    net.thread_graph_mode = True
    net._stop_token = object()
    net._graph_stop_event = threading.Event()
    net.graph_queues = [graph_queue]
    net.threads = [thread]
    net._progress_sampler = None
    net._progress_last_sample_active = None

    def _raise_drain_timeout(timeout=60.0):
        raise RuntimeError("drain timed out")

    net._wait_threaded_graph_idle = _raise_drain_timeout

    with pytest.raises(RuntimeError, match="Timed out joining Aspen threaded graph workers"):
        net.stop(wait=True)

    assert thread.join_timeouts
    assert thread.join_timeouts[0] is not None
    assert graph_queue.closed is True
    assert not hasattr(net, "graph_queues")
