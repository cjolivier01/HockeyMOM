import importlib.util
import json
import os
import sys
import tempfile
from types import ModuleType


def _load_hmlib_config_light():
    """Load hmlib/config.py without importing the full hmlib package.

    Creates stub modules in sys.modules for 'hmlib' and 'hmlib.bbox.box_functions'
    so that config.py can import without triggering heavy dependencies.
    """
    # Prepare stub hmlib module and its submodules
    hm_mod = ModuleType("hmlib")
    # Point __file__ to the real package path so ROOT_DIR resolves correctly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hm_path = os.path.join(repo_root, "hmlib", "__init__.py")
    hm_mod.__file__ = hm_path

    bbox_mod = ModuleType("hmlib.bbox")
    box_funcs_mod = ModuleType("hmlib.bbox.box_functions")

    def _scale_bbox_with_constraints(**kwargs):  # pragma: no cover - not used in tests
        return [0, 0, 0, 0]

    box_funcs_mod.scale_bbox_with_constraints = _scale_bbox_with_constraints

    sys.modules.setdefault("hmlib", hm_mod)
    sys.modules.setdefault("hmlib.bbox", bbox_mod)
    sys.modules.setdefault("hmlib.bbox.box_functions", box_funcs_mod)

    # Provide a very small YAML stub using JSON for the tests
    yaml_stub = ModuleType("yaml")

    def _safe_load(stream):
        return json.loads(stream.read())

    def _dump(data, stream=None, sort_keys=False):
        s = json.dumps(data)
        if stream is None:
            return s
        stream.write(s)

    yaml_stub.safe_load = _safe_load  # type: ignore[attr-defined]
    yaml_stub.dump = _dump  # type: ignore[attr-defined]
    sys.modules.setdefault("yaml", yaml_stub)

    config_path = os.path.join(repo_root, "hmlib", "config.py")
    spec = importlib.util.spec_from_file_location("hmlib.config", config_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader, "Failed to create import spec for hmlib.config"
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def should_merge_yaml_files_ordered():
    cfg = _load_hmlib_config_light()

    a = {
        "camera": {"name": "CamA"},
        "game": {"name": "G1"},
        "aspen": {
            "inference_pipeline": [{"type": "LoadImageFromFile"}],
        },
    }
    b = {
        "camera": {"name": "CamB"},
        "game": {"phase": "regular"},
        "aspen": {
            "video_out_pipeline": [{"type": "HmImageOverlays"}],
        },
    }

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "a.yaml")
        p2 = os.path.join(td, "b.yaml")
        with open(p1, "w") as f:
            f.write(json.dumps(a))
        with open(p2, "w") as f:
            f.write(json.dumps(b))

        merged = cfg.load_yaml_files_ordered([p1, p2])

    assert merged["camera"]["name"] == "CamB"
    assert merged["game"]["name"] == "G1"
    assert merged["game"]["phase"] == "regular"
    assert merged["aspen"]["inference_pipeline"][0]["type"] == "LoadImageFromFile"
    assert merged["aspen"]["video_out_pipeline"][0]["type"] == "HmImageOverlays"


def should_merge_aspen_namespace_mock():
    cfg = _load_hmlib_config_light()
    # Minimal base + aspen graph as JSON-y YAML files
    base = {"camera": {"name": "BaseCam"}, "game": {"name": "Base"}}
    graph = {"aspen": {"trunks": {"image_prep": {"class": "X", "depends": [], "params": {}}}}}

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "base.yaml")
        p2 = os.path.join(td, "graph.yaml")
        with open(p1, "w") as f:
            f.write(json.dumps(base))
        with open(p2, "w") as f:
            f.write(json.dumps(graph))
        merged = cfg.load_yaml_files_ordered([p1, p2])

    assert merged["camera"]["name"] == "BaseCam"
    assert merged["game"]["name"] == "Base"
    assert isinstance(merged.get("aspen"), dict)
    assert "trunks" in merged["aspen"]


def should_skip_private_game_config_when_requested():
    cfg = _load_hmlib_config_light()

    cfg.baseline_config = lambda root_dir=None: {}
    cfg.get_camera_config = lambda camera=None, root_dir=None: {}
    cfg.get_rink_config = lambda rink=None, root_dir=None: {}
    cfg.resolve_global_refs = lambda d: d

    private_calls = {"count": 0}

    def _load_config_file(
        root_dir=None, config_type=None, config_name=None, merge_into_config=None
    ):
        if config_type == "games":
            return {"game": {"name": "public-game"}}
        return {}

    def _get_game_config(game_id=None, root_dir=None):
        return {
            "game": {"name": "public-game"},
            "private_only": {"enabled": True},
        }

    def _get_game_config_private(game_id=None, merge_into_config=None):
        private_calls["count"] += 1
        return {"private_only": {"enabled": True}}

    cfg.load_config_file = _load_config_file
    cfg.get_game_config = _get_game_config
    cfg.get_game_config_private = _get_game_config_private

    merged = cfg.get_config(
        game_id="example-game",
        ignore_private_config=True,
        resolve_globals=False,
    )

    assert private_calls["count"] == 0
    assert merged.get("game", {}).get("name") == "public-game"
    assert "private_only" not in merged
