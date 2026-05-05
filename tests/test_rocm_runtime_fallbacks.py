import types
import unittest
from unittest import mock

from hmlib.utils.hockeymom_compat import (
    HOCKEYMOM_AVAILABLE,
    hockeymom_error_message,
    require_hockeymom,
)


class RocmRuntimeFallbacksTest(unittest.TestCase):
    def test_hmtrack_forces_rocm_safe_defaults(self):
        import hmlib.cli.hmtrack as hmtrack_cli

        args = types.SimpleNamespace(
            explicit_arg_names=set(),
            tracker_backend=None,
            python_blender=False,
        )
        game_config = {
            "stitching": {"python_blender": False},
            "aspen": {
                "plugins": {
                    "stitching": {"params": {"python_blender": False}},
                    "camera_controller": {"enabled": True},
                    "play_tracker": {"enabled": True},
                    "save_camera": {"enabled": True},
                    "rgb_stats_check": {"enabled": True},
                }
            },
        }

        with mock.patch.object(hmtrack_cli, "is_rocm_backend", return_value=True):
            hmtrack_cli._apply_rocm_runtime_defaults(args, game_config)

        self.assertEqual(args.tracker_backend, "static_bytetrack")
        self.assertTrue(args.python_blender)
        self.assertTrue(game_config["stitching"]["python_blender"])
        self.assertTrue(game_config["aspen"]["plugins"]["stitching"]["params"]["python_blender"])
        self.assertFalse(game_config["aspen"]["plugins"]["camera_controller"]["enabled"])
        self.assertFalse(game_config["aspen"]["plugins"]["play_tracker"]["enabled"])
        self.assertFalse(game_config["aspen"]["plugins"]["save_camera"]["enabled"])
        self.assertFalse(game_config["aspen"]["plugins"]["rgb_stats_check"]["enabled"])

    def test_hmtrack_rocm_defaults_do_not_create_missing_plugin_stubs(self):
        import hmlib.cli.hmtrack as hmtrack_cli

        args = types.SimpleNamespace(
            explicit_arg_names=set(),
            tracker_backend=None,
            python_blender=False,
        )
        game_config = {
            "stitching": {"python_blender": False},
            "aspen": {
                "plugins": {
                    "stitching": {
                        "class": "hmlib.aspen.plugins.stitching_plugin.StitchingPlugin",
                        "params": {"python_blender": False},
                    },
                    "camera_controller": {
                        "class": "hmlib.aspen.plugins.camera_controller_plugin.CameraControllerPlugin",
                        "enabled": True,
                    },
                }
            },
        }

        with mock.patch.object(hmtrack_cli, "is_rocm_backend", return_value=True):
            hmtrack_cli._apply_rocm_runtime_defaults(args, game_config)

        plugins_cfg = game_config["aspen"]["plugins"]
        self.assertFalse(plugins_cfg["camera_controller"]["enabled"])
        self.assertNotIn("play_tracker", plugins_cfg)
        self.assertNotIn("save_camera", plugins_cfg)
        self.assertNotIn("rgb_stats_check", plugins_cfg)

    def test_stitch_forces_python_blender_on_rocm(self):
        import hmlib.cli.stitch as stitch_cli

        args = types.SimpleNamespace(explicit_arg_names=set(), python_blender=False)
        game_config = {
            "stitching": {"python_blender": False},
            "aspen": {"plugins": {"stitching": {"params": {"python_blender": False}}}},
        }

        with mock.patch.object(stitch_cli, "is_rocm_backend", return_value=True):
            enabled = stitch_cli._apply_rocm_stitch_defaults(
                args,
                game_config,
                python_blender=False,
            )

        self.assertTrue(enabled)
        self.assertTrue(args.python_blender)
        self.assertTrue(game_config["stitching"]["python_blender"])
        self.assertTrue(game_config["aspen"]["plugins"]["stitching"]["params"]["python_blender"])

    def test_missing_native_extension_reports_explicit_error(self):
        if HOCKEYMOM_AVAILABLE:
            self.skipTest("hockeymom native extension is available in this environment")
        with self.assertRaisesRegex(RuntimeError, "requires the hockeymom native extension"):
            require_hockeymom("native panorama stitching")
        self.assertTrue(hockeymom_error_message())

    def test_end_to_end_disables_cpp_bytetrack_on_rocm(self):
        import importlib.util

        if importlib.util.find_spec("mmcv") is None:
            self.skipTest("mmcv is not available in this environment")

        import hmlib.models.end_to_end_plugin as end_to_end_plugin

        with mock.patch.object(end_to_end_plugin, "is_rocm_backend", return_value=True):
            self.assertFalse(end_to_end_plugin._should_use_cpp_bytetrack(True))


if __name__ == "__main__":
    unittest.main()
