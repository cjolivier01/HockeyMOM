import pytest

from hmlib.aspen import AspenNet


def should_require_join_plugin_for_any_fanin():
    graph = {
        "plugins": {
            "tracker": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": [],
                "params": {"name": "tracker"},
            },
            "camera_controller": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["tracker"],
                "params": {"name": "camera_controller"},
            },
            "play_tracker": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["tracker", "camera_controller"],
                "params": {"name": "play_tracker"},
            },
        }
    }

    with pytest.raises(ValueError) as excinfo:
        AspenNet("multipath", graph)

    msg = str(excinfo.value)
    assert "fan-in" in msg.lower()
    assert "joinplugin" in msg.lower()
    assert "tracker" in msg
    assert "play_tracker" in msg
    assert "camera_controller" in msg


def should_allow_join_plugin_to_merge_parallel_branches():
    graph = {
        "plugins": {
            "tracker": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": [],
                "params": {"name": "tracker"},
            },
            "play_tracker": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["tracker"],
                "params": {"name": "play_tracker"},
            },
            "pose": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["tracker"],
                "params": {"name": "pose"},
            },
            "join": {
                "class": "hmlib.aspen.plugins.join_plugin.JoinPlugin",
                "depends": ["play_tracker", "pose"],
                "params": {"required_plugins": ["play_tracker", "pose"], "output_key": "joined"},
            },
            "downstream": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.RecordPlugin",
                "depends": ["join"],
                "params": {"name": "downstream"},
            },
        }
    }

    net = AspenNet("multipath_join_ok", graph)
    net({})


def should_join_merge_outputs_and_reject_duplicate_keys():
    ok_graph = {
        "plugins": {
            "a": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.KeyPlugin",
                "depends": [],
                "params": {"outputs": {"ka": 1}},
            },
            "b": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.KeyPlugin",
                "depends": [],
                "params": {"outputs": {"kb": 2}},
            },
            "join": {
                "class": "hmlib.aspen.plugins.join_plugin.JoinPlugin",
                "depends": ["a", "b"],
                "params": {"required_plugins": ["a", "b"], "output_key": "joined"},
            },
        }
    }
    net = AspenNet("join_merge_ok", ok_graph)
    out = net({})
    assert out["joined"] == {"ka": 1, "kb": 2}

    dup_graph = {
        "plugins": {
            "a": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.KeyPlugin",
                "depends": [],
                "params": {"outputs": {"k": 1}},
            },
            "b": {
                "class": "hmlib.aspen.plugins.test_threaded_graph_plugin.KeyPlugin",
                "depends": [],
                "params": {"outputs": {"k": 2}},
            },
            "join": {
                "class": "hmlib.aspen.plugins.join_plugin.JoinPlugin",
                "depends": ["a", "b"],
                "params": {"required_plugins": ["a", "b"], "output_key": "joined"},
            },
        }
    }
    net2 = AspenNet("join_merge_dup", dup_graph)
    with pytest.raises(ValueError) as excinfo:
        net2({})
    assert "duplicate" in str(excinfo.value).lower()
