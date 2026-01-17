from __future__ import absolute_import, division, print_function

import argparse
import logging
from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import yaml

from hmlib.config import get_game_config_private, get_nested_value, set_nested_value

logger = logging.getLogger(__name__)


_SKIP_CONFIG_VALUE = object()
_MISSING_ARG = object()


def _get_arg_value(args: Any, name: str) -> Any:
    if isinstance(args, dict):
        return args.get(name, _MISSING_ARG)
    return getattr(args, name, _MISSING_ARG)


def _has_nested_key(dct: Dict[str, Any], key_str: str) -> bool:
    cur: Any = dct
    for key in key_str.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _coerce_aspen_queue_size(value: Any) -> Any:
    try:
        return max(1, int(value))
    except Exception:
        return _SKIP_CONFIG_VALUE


def _coerce_aspen_max_concurrent(value: Any) -> Any:
    try:
        return max(1, int(value))
    except Exception:
        return _SKIP_CONFIG_VALUE


def _debug_to_play_tracker(value: Any) -> Any:
    try:
        if isinstance(value, str):
            value = int(value)
        return True if int(value) >= 1 else _SKIP_CONFIG_VALUE
    except Exception:
        return _SKIP_CONFIG_VALUE


def _get_disable_progress_bar() -> bool:
    import os

    return bool(os.environ.get("SLURM_JOBID", ""))


def copy_opts(src: object, dest: object, parser: argparse.ArgumentParser):
    """Copy known CLI options from one namespace-like object to another.

    Uses the provided parser to discover option names, then copies any
    attributes with matching names from ``src`` to ``dest``.

    @param src: Source object (typically parsed args).
    @param dest: Destination object to mutate.
    @param parser: Parser used to determine which attributes to copy.
    @return: The updated ``dest`` object.
    """
    fake_parsed = parser.parse_known_args()
    item_keys = sorted(fake_parsed[0].__dict__.keys())
    for item_name in item_keys:
        if hasattr(src, item_name):
            setattr(dest, item_name, getattr(src, item_name))
    return dest


class hm_opts(object):
    """Shared command-line options used by most HockeyMOM tools.

    The :meth:`parser` static method populates an :class:`argparse.ArgumentParser`
    with all common flags (I/O, caching, profiling, Aspen, ONNX/TensorRT, UI, etc.),
    and instances of :class:`hm_opts` hold the parsed values.

    @see @ref hmlib.utils.profiler.HmProfiler "HmProfiler" for profiling options.
    @see @ref hmlib.utils.progress_bar.ProgressBar "ProgressBar" for progress UI controls.
    """

    def __init__(self, parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        self._parser: argparse.ArgumentParser = self.parser(parser)

    @staticmethod
    def parser(parser: argparse.ArgumentParser = None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cameraname",
            "--camera",
            dest="camera_name",
            default=None,
            type=str,
            help="Cameraname",
        )
        parser.add_argument(
            "--gpus", default="0,1,2", help="-1 for CPU, use comma for multiple gpus"
        )
        parser.add_argument("--debug", default=0, type=int, help="debug level")
        parser.add_argument(
            "--checkerboard-input",
            dest="checkerboard_input",
            action="store_true",
            help=(
                "Replace input video frames with a synthetic checkerboard pattern "
                "and emit per-frame RGB statistics for debugging."
            ),
        )
        audit = parser.add_argument_group(
            "audit",
            "Pipeline audit (per-plugin frame hashing / comparison)",
        )
        audit.add_argument(
            "--audit-dir",
            dest="audit_dir",
            type=str,
            default=None,
            help=(
                "Directory to write per-plugin frame hashes (audit.jsonl). "
                "When --audit-reference-dir is provided, hashes are compared and mismatches "
                "are written to mismatches.jsonl."
            ),
        )
        audit.add_argument(
            "--audit-reference-dir",
            dest="audit_reference_dir",
            type=str,
            default=None,
            help="Directory containing a reference audit.jsonl to compare against.",
        )
        audit.add_argument(
            "--audit-plugins",
            dest="audit_plugins",
            type=str,
            default=None,
            help="Comma-separated Aspen plugin names to audit (default: all plugins).",
        )
        audit.add_argument(
            "--audit-dump-images",
            dest="audit_dump_images",
            action="store_true",
            help="Dump PNG images for all captured audit tensors (can be large).",
        )
        audit.add_argument(
            "--audit-fail-fast",
            dest="audit_fail_fast",
            type=int,
            default=1,
            choices=[0, 1],
            help="Stop execution on the first audit mismatch (default: 1).",
        )
        parser.add_argument(
            "--crop-play-box",
            default=None,
            type=int,
            help="Crop to play area only",
        )
        parser.add_argument(
            "--no-crop",
            dest="no_crop",
            action="store_true",
            help="Disable camera/output cropping in the video pipeline",
        )
        parser.add_argument(
            "--end-zones",
            action="store_true",
            help="Enable end-zone camera usage when available",
        )
        parser.add_argument(
            "--unsharp-mask",
            default=None,
            type=float,
            help="Apply unsharp masking to frame (good for blurry LiveBarn footage)",
        )
        # Input color adjustments (applied in inference pipeline via HmImageColorAdjust)
        parser.add_argument(
            "--white-balance",
            dest="white_balance",
            nargs=3,
            type=float,
            default=None,
            metavar=("R_GAIN", "G_GAIN", "B_GAIN"),
            help="Per-channel RGB gains for white balance (e.g., 1.05 1.0 0.95)",
        )
        parser.add_argument(
            "--white-balance-k",
            "--white-balance-temp",
            dest="white_balance_k",
            type=str,
            default=None,
            help="White balance correlated color temperature (e.g., 3500k, 4700k, 6500k)",
        )
        parser.add_argument(
            "--color-brightness",
            dest="color_brightness",
            type=float,
            default=None,
            help="Brightness multiplier (>1 brighter). No-op if omitted.",
        )
        parser.add_argument(
            "--color-contrast",
            dest="color_contrast",
            type=float,
            default=None,
            help="Contrast factor (>1 more contrast). No-op if omitted.",
        )
        parser.add_argument(
            "--color-gamma",
            dest="color_gamma",
            type=float,
            default=None,
            help="Gamma exponent (>1 darker). No-op if omitted.",
        )

        #
        # Data I/O
        #
        io = parser.add_argument_group("Data I/O")
        # Video input/output
        io.add_argument(
            "--input-video",
            type=str,
            default=None,
            help="Input video file(s)",
        )
        io.add_argument(
            "--output-video",
            type=str,
            default=None,
            help="The output video file name",
        )
        io.add_argument(
            "--no-save-video",
            "--no_save_video",
            dest="no_save_video",
            action="store_true",
            help="Don't save the output video",
        )
        io.add_argument(
            "--save-frame-dir",
            type=str,
            default=None,
            help="Directory to save output frames as PNG files",
        )
        io.add_argument(
            "--audio-only",
            action="store_true",
            help="Only transfer the audio",
        )
        io.add_argument(
            "--no-audio",
            action="store_true",
            help="Skip copying audio to the rendered video",
        )
        io.add_argument(
            "--deploy-dir",
            dest="deploy_dir",
            type=str,
            default=None,
            help=(
                "Optional directory to deploy output artifacts (video/CSVs) to on completion. "
                "Full runs default to deploying into the game directory when this is omitted; "
                "short -t runs only deploy when --deploy-dir is set."
            ),
        )
        # Feature caching flags moved to their own group
        io.add_argument(
            "--save-camera-data",
            action="store_true",
            help="Save tracking data to camera.csv",
        )
        io.add_argument(
            "--input-tracking-data",
            dest="input_tracking_data",
            type=str,
            default=None,
            help="Path to a precomputed tracking CSV to load instead of running the tracker.",
        )
        io.add_argument(
            "--input-detection-data",
            dest="input_detection_data",
            type=str,
            default=None,
            help="Path to a precomputed detections CSV to load instead of running the detector.",
        )
        io.add_argument(
            "--input-pose-data",
            dest="input_pose_data",
            type=str,
            default=None,
            help="Path to a precomputed pose CSV to load instead of running pose inference.",
        )
        io.add_argument(
            "--save-pose-data",
            dest="save_pose_data",
            action="store_true",
            help="Enable saving pose results to pose.csv via Aspen SavePosePlugin (when configured).",
        )

        #
        # Visualization & Plotting
        #
        plot = parser.add_argument_group("Visualization & Plotting")
        plot.add_argument(
            "--plot-tracking",
            action="store_true",
            help="Plot individual tracking overlays (circles by default)",
        )
        plot.add_argument(
            "--no-plot-tracking-circles",
            dest="no_plot_tracking_circles",
            action="store_true",
            help="Disable tracking circles and draw bounding boxes instead.",
        )
        plot.add_argument("--plot-ice-mask", action="store_true", help="Plot the ice mask")
        plot.add_argument(
            "--plot-trajectories", action="store_true", help="Plot individual track trajectories"
        )
        plot.add_argument(
            "--plot-jersey-numbers", action="store_true", help="Plot individual jersey numbers"
        )
        plot.add_argument(
            "--plot-actions", action="store_true", help="Plot action labels per tracked player"
        )
        plot.add_argument("--plot-pose", action="store_true", help="Plot individual pose skeletons")
        plot.add_argument(
            "--plot-overhead-rink",
            action="store_true",
            help="Draw an overhead rink minimap with player positions",
        )
        plot.add_argument(
            "--plot-all-detections",
            type=float,
            default=None,
            help="Plot all detections above this given accuracy",
        )
        plot.add_argument(
            "--plot-moving-boxes",
            action="store_true",
            help="Plot moving camera tracking boxes",
        )
        # Pose visualization tuning
        plot.add_argument(
            "--kpt-thr", type=float, default=0.3, help="Keypoint score threshold for overlay"
        )
        plot.add_argument(
            "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold for overlay"
        )
        plot.add_argument("--radius", type=int, default=4, help="Keypoint radius for overlay")
        plot.add_argument("--thickness", type=int, default=1, help="Link thickness for overlay")

        #
        # Profiling
        #
        prof = parser.add_argument_group("Profiling")
        prof.add_argument(
            "--profile",
            dest="profile",
            action="store_true",
            help="Enable PyTorch Perfetto/Chrome profiler and export trace JSON",
        )
        prof.add_argument(
            "--profile-dir",
            dest="profile_dir",
            type=str,
            default=".",
            help="Directory to write profiler traces (defaults under output_workdirs/<game_id>/profiler)",
        )
        prof.add_argument(
            "--profile-record-shapes",
            dest="profile_record_shapes",
            action="store_true",
            help="Record tensor shapes in profiler (adds overhead)",
        )
        prof.add_argument(
            "--profile-memory",
            dest="profile_memory",
            action="store_true",
            help="Track memory in profiler (adds overhead)",
        )
        prof.add_argument(
            "--profile-with-stack",
            dest="profile_with_stack",
            action="store_true",
            default=None,
            help="Capture Python stack traces in profiler events (default when --profile is set; adds overhead)",
        )
        prof.add_argument(
            "--no-profile-stack",
            "--profile-no-stack",
            dest="profile_with_stack",
            action="store_false",
            default=None,
            help="Disable stack trace capture for profiling runs.",
        )
        prof.add_argument(
            "--profile-export-per-iter",
            dest="profile_export_per_iter",
            action="store_true",
            help="Export one trace per iteration (large runs; adds I/O)",
        )
        prof.add_argument(
            "--profile-step",
            dest="profile_step",
            type=int,
            default=None,
            help="Start profiler at this 1-based iteration index (default: start immediately)",
        )
        prof.add_argument(
            "--profile-step-count",
            dest="profile_step_count",
            type=int,
            default=0,
            help="Number of iterations to profile once started (default: 1)",
        )
        prof.add_argument(
            "--py-trace-out",
            dest="py_trace_out",
            type=str,
            default=None,
            help="Optional Python cProfile output file (.pstats or .txt)",
        )

        #
        # Camera Controller
        #
        cam_ctrl = parser.add_argument_group("Camera Controller")
        cam_ctrl.add_argument(
            "--camera-controller",
            type=str,
            choices=["rule", "transformer", "gpt"],
            default="rule",
            help="Select camera controller: rule-based PlayTracker, transformer, or GPT model",
        )
        cam_ctrl.add_argument(
            "--camera-model",
            type=str,
            default=None,
            help=(
                "Path to camera model checkpoint (.pt) produced by camtrain.py "
                "(transformer) or camgpt_train.py (gpt)"
            ),
        )
        cam_ctrl.add_argument(
            "--camera-window",
            type=int,
            default=8,
            help="Temporal window length to feed the transformer controller",
        )
        #
        # TensorRT options (Detector)
        #
        trt_det = parser.add_argument_group("TensorRT Detector")
        trt_det.add_argument(
            "--detector-trt-enable",
            dest="detector_trt_enable",
            action="store_true",
            help="Enable TensorRT for detector (backbone+neck). Builds engine on first run if needed.",
        )
        trt_det.add_argument(
            "--detector-trt-engine",
            dest="detector_trt_engine",
            type=str,
            default=None,
            help="Path to save/load the detector TensorRT engine (defaults under output_workdirs/<GAME_ID>/detector.engine).",
        )
        trt_det.add_argument(
            "--detector-trt-fp16",
            dest="detector_trt_fp16",
            action="store_true",
            help="Build TensorRT detector engine in FP16 mode if supported.",
        )
        trt_det.add_argument(
            "--detector-trt-int8",
            dest="detector_trt_int8",
            action="store_true",
            help="Build TensorRT detector engine in INT8 mode with calibration frames.",
        )
        trt_det.add_argument(
            "--detector-trt-calib-frames",
            dest="detector_trt_calib_frames",
            type=int,
            default=200,
            help="Number of early frames to collect for TensorRT INT8 calibration (default: 200).",
        )
        trt_det.add_argument(
            "--detector-trt-force-build",
            dest="detector_trt_force_build",
            action="store_true",
            help="Force rebuilding the detector TensorRT engine even if it exists.",
        )
        trt_det.add_argument(
            "--detector-static-detections",
            dest="detector_static_detections",
            action="store_true",
            help="Enable fixed-shape detector head outputs to avoid dynamic-mask stream sync.",
        )
        trt_det.add_argument(
            "--detector-static-max-detections",
            dest="detector_static_max_detections",
            type=int,
            default=0,
            help="Max detections to keep when static detections are enabled (default: use model test_cfg).",
        )
        trt_det.add_argument(
            "--detector-nms-backend",
            dest="detector_nms_backend",
            type=str,
            default="trt",
            choices=["trt", "torchvision", "head"],
            help="NMS backend when using TensorRT detector: "
            "'trt' (TensorRT batched NMS plugin), "
            "'torchvision' (torchvision.ops.nms per class), or "
            "'head' (use bbox head's original NMS).",
        )
        trt_det.add_argument(
            "--detector-nms-test",
            dest="detector_nms_test",
            action="store_true",
            help="Debug mode: when using TensorRT detector, run both TensorRT batched NMS "
            "and torchvision NMS and log basic differences per frame.",
        )
        trt_det.add_argument(
            "--detector-trt-nms-plugin",
            dest="detector_trt_nms_plugin",
            type=str,
            default="efficient",
            choices=["batched", "efficient"],
            help="TensorRT NMS plugin to use for detector path when backend is 'trt': "
            "'efficient' (EfficientNMS_TRT, default) or 'batched' (BatchedNMSDynamic_TRT).",
        )
        #
        # TensorRT options (Pose)
        #
        trt_pose = parser.add_argument_group("TensorRT Pose")
        trt_pose.add_argument(
            "--pose-trt-enable",
            dest="pose_trt_enable",
            action="store_true",
            help="Enable TensorRT for pose (backbone+neck). Builds engine on first run if needed.",
        )
        trt_pose.add_argument(
            "--pose-trt-engine",
            dest="pose_trt_engine",
            type=str,
            default=None,
            help="Path to save/load the pose TensorRT engine (defaults under output_workdirs/<GAME_ID>/pose.engine).",
        )
        trt_pose.add_argument(
            "--pose-trt-fp16",
            dest="pose_trt_fp16",
            action="store_true",
            help="Build TensorRT pose engine in FP16 mode if supported.",
        )
        trt_pose.add_argument(
            "--pose-trt-int8",
            dest="pose_trt_int8",
            action="store_true",
            help="Build TensorRT pose engine in INT8 mode with calibration frames.",
        )
        trt_pose.add_argument(
            "--pose-trt-calib-frames",
            dest="pose_trt_calib_frames",
            type=int,
            default=200,
            help="Number of early frames to collect for TensorRT INT8 calibration for pose (default: 200).",
        )
        trt_pose.add_argument(
            "--pose-trt-force-build",
            dest="pose_trt_force_build",
            action="store_true",
            help="Force rebuilding the pose TensorRT engine even if it exists.",
        )

        #
        # ONNX options (Detector)
        #
        onnx_det = parser.add_argument_group("ONNX Detector")
        onnx_det.add_argument(
            "--detector-onnx",
            dest="detector_onnx_path",
            type=str,
            default=None,
            help=(
                "Export the detector to ONNX at this path and run inference with ONNX Runtime. "
                "If a path is not provided here, a default under output_workdirs/<GAME_ID>/detector.onnx is used."
            ),
        )
        onnx_det.add_argument(
            "--detector-onnx-enable",
            dest="detector_onnx_enable",
            action="store_true",
            help=(
                "Enable ONNX Runtime detector inference. If --detector-onnx is provided, enablement is implied."
            ),
        )
        onnx_det.add_argument(
            "--detector-onnx-quantize-int8",
            dest="detector_onnx_quantize_int8",
            action="store_true",
            help=(
                "After exporting the float32 model, quantize to INT8. "
                "Calibration samples are gathered on-the-fly from early frames."
            ),
        )
        onnx_det.add_argument(
            "--detector-onnx-calib-frames",
            dest="detector_onnx_calib_frames",
            type=int,
            default=200,
            help="Number of frames to collect for INT8 calibration (default: 200)",
        )
        onnx_det.add_argument(
            "--detector-onnx-force-export",
            dest="detector_onnx_force_export",
            action="store_true",
            help="Force re-exporting ONNX even if the file already exists",
        )

        #
        # ONNX options (Pose)
        #
        onnx_pose = parser.add_argument_group("ONNX Pose")
        onnx_pose.add_argument(
            "--pose-onnx",
            dest="pose_onnx_path",
            type=str,
            default=None,
            help=(
                "Export the pose model's feature extractor (backbone+neck) to ONNX and run with ONNX Runtime. "
                "If a path is not provided, a default under output_workdirs/<GAME_ID>/pose.onnx is used."
            ),
        )
        onnx_pose.add_argument(
            "--pose-onnx-enable",
            dest="pose_onnx_enable",
            action="store_true",
            help=(
                "Enable ONNX Runtime for pose (backbone+neck). If --pose-onnx is provided, enablement is implied."
            ),
        )
        onnx_pose.add_argument(
            "--pose-onnx-quantize-int8",
            dest="pose_onnx_quantize_int8",
            action="store_true",
            help=(
                "After exporting the float32 pose model, quantize to INT8 using calibration frames."
            ),
        )
        onnx_pose.add_argument(
            "--pose-onnx-calib-frames",
            dest="pose_onnx_calib_frames",
            type=int,
            default=200,
            help="Number of frames to collect for pose INT8 calibration (default: 200)",
        )
        onnx_pose.add_argument(
            "--pose-onnx-force-export",
            dest="pose_onnx_force_export",
            action="store_true",
            help="Force re-exporting ONNX for pose even if the file already exists",
        )
        #
        # Tracker options
        #
        tracker = parser.add_argument_group("Tracker")
        tracker.add_argument(
            "--tracker-backend",
            dest="tracker_backend",
            type=str,
            choices=["hm", "static_bytetrack"],
            default=None,
            help=(
                "Select tracking backend: 'hm' (default, HmTracker) or "
                "'static_bytetrack' (CUDA static ByteTrack with fixed max_detections/max_tracks)."
            ),
        )
        tracker.add_argument(
            "--tracker-max-detections",
            dest="tracker_max_detections",
            type=int,
            default=256,
            help=(
                "Maximum detections per frame passed to the static ByteTrack tracker "
                "when --tracker-backend=static_bytetrack is used."
            ),
        )
        tracker.add_argument(
            "--tracker-max-tracks",
            dest="tracker_max_tracks",
            type=int,
            default=256,
            help=(
                "Maximum active tracks maintained by the static ByteTrack tracker "
                "when --tracker-backend=static_bytetrack is used."
            ),
        )
        tracker.add_argument(
            "--tracker-device",
            dest="tracker_device",
            type=str,
            default=None,
            help=(
                "Optional device string for the static ByteTrack tracker "
                "(e.g., 'cuda:0'); defaults to the main detection device."
            ),
        )
        parser.add_argument(
            "--deterministic",
            default=0,
            type=int,
            help="Whether we should try to be deterministic",
        )
        # Identity
        parser.add_argument(
            "--team",
            default=None,
            type=str,
            help="The primary team that represents the configuration file",
        )
        parser.add_argument(
            "--season",
            default=None,
            type=str,
            help="Season (if not the current)",
        )
        parser.add_argument(
            "--game-id",
            default=None,
            type=str,
            help="Game ID",
        )
        parser.add_argument(
            "--serial",
            default=0,
            type=int,
            help="Serial execution of entire pipeline",
        )
        # stitching
        parser.add_argument(
            "--cache-size",
            type=int,
            default=2,
            help="cache size for GPU stream async operations",
        )
        async_group = parser.add_mutually_exclusive_group()
        async_group.add_argument(
            "--no-async-dataset",
            dest="no_async_dataset",
            action="store_true",
            help="Disable async dataset loading and use synchronous video I/O.",
        )
        parser.add_argument(
            "--no-cuda-streams",
            action="store_true",
            help="Don't use CUDA streams",
        )
        aspen_thread_group = parser.add_mutually_exclusive_group()
        aspen_thread_group.add_argument(
            "--aspen-threaded",
            dest="aspen_threaded",
            action="store_true",
            help="Run Aspen plugins in threaded pipeline mode",
        )
        aspen_thread_group.add_argument(
            "--no-aspen-threaded",
            dest="aspen_threaded",
            action="store_false",
            help="Disable threaded Aspen pipeline mode",
        )
        parser.set_defaults(aspen_threaded=None)
        aspen_graph_group = parser.add_mutually_exclusive_group()
        aspen_graph_group.add_argument(
            "--aspen-thread-graph",
            dest="aspen_thread_graph",
            action="store_true",
            help="Run threaded Aspen plugins in graph scheduling mode",
        )
        aspen_graph_group.add_argument(
            "--no-aspen-thread-graph",
            dest="aspen_thread_graph",
            action="store_false",
            help="Use linear scheduling for threaded Aspen plugins",
        )
        parser.set_defaults(aspen_thread_graph=None)
        parser.add_argument(
            "--aspen-thread-queue-size",
            dest="aspen_thread_queue_size",
            type=int,
            default=None,
            help="Queue size between threaded Aspen plugins (defaults to 1)",
        )
        parser.add_argument(
            "--aspen-max-concurrent",
            dest="aspen_max_concurrent",
            type=int,
            default=None,
            help="Max concurrent frames in threaded Aspen pipeline (defaults to 3)",
        )
        aspen_stream_group = parser.add_mutually_exclusive_group()
        aspen_stream_group.add_argument(
            "--aspen-thread-cuda-streams",
            dest="aspen_thread_cuda_streams",
            action="store_true",
            help="Give each threaded Aspen trunk its own CUDA stream",
        )
        aspen_stream_group.add_argument(
            "--no-aspen-thread-cuda-streams",
            dest="aspen_thread_cuda_streams",
            action="store_false",
            help="Disable per-trunk CUDA streams in threaded Aspen mode",
        )
        parser.set_defaults(aspen_thread_cuda_streams=None)
        aspen_stitch_group = parser.add_mutually_exclusive_group()
        aspen_stitch_group.add_argument(
            "--aspen-stitching",
            dest="aspen_stitching",
            action="store_true",
            help="Enable Aspen stitching plugin for multi-camera inputs",
        )
        aspen_stitch_group.add_argument(
            "--no-aspen-stitching",
            dest="aspen_stitching",
            action="store_false",
            help="Disable Aspen stitching plugin",
        )
        parser.set_defaults(aspen_stitching=None)
        profile_group = parser.add_mutually_exclusive_group()
        profile_group.add_argument(
            "--display-plugin-profile",
            dest="display_plugin_profile",
            action="store_true",
            help="Show per-plugin timing percentages in the progress table",
        )
        profile_group.add_argument(
            "--no-display-plugin-profile",
            dest="display_plugin_profile",
            action="store_false",
            help="Hide per-plugin timing percentages in the progress table",
        )
        parser.set_defaults(display_plugin_profile=None)
        graph_display_group = parser.add_mutually_exclusive_group()
        graph_display_group.add_argument(
            "--display-aspen-graph",
            dest="display_aspen_graph",
            action="store_true",
            help="Show AspenNet graph activity in the progress UI",
        )
        graph_display_group.add_argument(
            "--no-display-aspen-graph",
            dest="display_aspen_graph",
            action="store_false",
            help="Hide AspenNet graph activity in the progress UI",
        )
        parser.set_defaults(display_aspen_graph=None)
        parser.add_argument(
            "--fp16",
            default=False,
            action="store_true",
            help="show as processing",
        )
        parser.add_argument(
            "--fp16-stitch",
            default=False,
            action="store_true",
            help="Stitch images using fp16 (lower mem, but lower quality image output)",
        )
        parser.add_argument(
            "--show-image",
            "--show",
            dest="show_image",
            default=False,
            action="store_true",
            help="show as processing",
        )
        parser.add_argument(
            "--show-image-name",
            default="default",
            type=str,
            help="Name of the image to show, i.e. 'default', 'end_zones'",
        )
        parser.add_argument(
            "--show-scaled",
            type=float,
            default=None,
            help="scale showed image (ignored is --show-image is not specified)",
        )
        parser.add_argument(
            "--scoreboard-scale",
            dest="scoreboard_scale",
            type=float,
            default=None,
            help="Scale factor applied to the extracted scoreboard image (maps to rink.scoreboard.scoreboard_scale)",
        )
        parser.add_argument(
            "--ice-rink-inference-scale",
            "--ice-rink-mask-scale",
            dest="ice_rink_inference_scale",
            type=float,
            default=None,
            help="Downscale factor for ice rink segmentation (e.g., 0.5 doubles speed, 1.0 keeps original size)",
        )
        parser.add_argument(
            "--decoder",
            "--video-stream-decode-method",
            dest="video_stream_decode_method",
            default="auto",
            type=str,
            help=(
                "Video stream decode method [auto, cv2, ffmpeg, torchvision, torchaudio, "
                "gstreamer, pynvcodec]"
            ),
        )
        parser.add_argument(
            "--decoder-device",
            default="cuda",
            type=str,
            help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        )
        parser.add_argument(
            "--encoder-device",
            default=None,
            type=str,
            help="Video stream encode device [cpu, cude, cuda:0, etc.]",
        )
        parser.add_argument(
            "--video-encoder-backend",
            dest="video_encoder_backend",
            choices=["auto", "pyav", "ffmpeg", "raw"],
            default=None,
            help=(
                "Backend for PyNvVideoEncoder when using NVENC writers. "
                "Values: auto (use baseline.yaml / auto-detect), pyav, ffmpeg, raw. "
                "When provided, this overrides aspen.video_out.encoder_backend from baseline.yaml."
            ),
        )
        # parser.add_argument(
        #     "--encoder",
        #     "--video-stream-encode-method",
        #     dest="video_stream_encode_method",
        #     default="cv2",
        #     type=str,
        #     help="Video stream decode method [cv2, ffmpeg, torchvision, tochaudio]",
        # )
        parser.add_argument(
            "-o",
            "--output",
            dest="output_file",
            type=str,
            default=None,
            help="Output file",
        )
        parser.add_argument(
            "--output-fps",
            dest="output_fps",
            type=float,
            default=None,
            help="Output frames per second",
        )
        parser.add_argument(
            "--output-width",
            dest="output_width",
            type=int,
            default=None,
            help="Resize the rendered output video to this width (keeps aspect ratio).",
        )
        parser.add_argument(
            "--lfo",
            "--left-frame-offset",
            dest="lfo",
            type=float,
            default=None,
            help="Offset for left video startig point (first supplied video)",
        )
        parser.add_argument(
            "--plot-frame-number",
            type=int,
            default=0,
            help="Plot frame number",
        )
        parser.add_argument(
            "--plot-frame-time",
            type=int,
            default=0,
            help="Plot frame time",
        )
        parser.add_argument(
            "--rfo",
            "--right-frame-offset",
            dest="rfo",
            type=float,
            default=None,
            help="Offset for right video startiog point (second supplied video)",
        )
        parser.add_argument(
            "--start-frame-offset",
            default=0,
            help="General start frame the video reading (after other offsets are applied)",
        )
        parser.add_argument(
            "--project-file",
            "--project_file",
            dest="project_file",
            default="hm_project.pto",
            type=str,
            help="Use project file as input to stitcher",
        )
        parser.add_argument(
            "--start-frame", type=int, default=0, help="first frame number to process"
        )
        parser.add_argument(
            "-s",
            "--start-time",
            "--start-frame-time",
            dest="start_frame_time",
            type=str,
            default=None,
            help="Start at this time in video stream",
        )
        parser.add_argument(
            "--stitch-frame-time",
            type=str,
            default=None,
            help="Use frame at this timestamp for stitching (HH:MM:SS.ssss)",
        )
        parser.add_argument(
            "--max-levels",
            "--max_levels",
            "--max-blend-levels",
            "--max_blend_levels",
            dest="max_blend_levels",
            type=int,
            default=None,
            help=(
                "Maximum Laplacian blend pyramid levels for stitching "
                "(applies to laplacian / gpu Laplacian modes; hard-seam uses 0)"
            ),
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="maximum number of frames to process",
        )
        parser.add_argument(
            "-t",
            "--max-time",
            dest="max_time",
            type=str,
            default=None,
            help="Maximum amount of time to process",
        )
        parser.add_argument(
            "--video_dir",
            default=None,
            type=str,
            help="Video directory to find 'left.mp4' and 'right.mp4'",
        )
        parser.add_argument(
            "--minimize-blend",
            type=int,
            default=None,
            choices=[0, 1],
            help="Minimize blending compute to only blend (mostly) overlapping portions of frames",
        )
        parser.add_argument(
            "--blend-mode",
            "--blend_mode",
            default="laplacian",
            type=str,
            help="Stitching blend mode (multiblend|laplacian|gpu-hard-seam)",
        )
        parser.add_argument(
            "--skip_final_video_save",
            "--skip-final-video-save",
            action="store_true",
            default=None,
            help="Don't save the output video frames",
        )
        parser.add_argument(
            "--save_stitched",
            "--save-stitched",
            action="store_true",
            help="Don't save the output video",
        )
        parser.add_argument(
            "--stitch-auto-adjust-exposure",
            type=int,
            default=1,
            help="Auto-adjust exposure when stitching",
        )
        parser.add_argument(
            "--no-minimize-blend",
            action="store_true",
            help="Don't minimize blending to the overlapped portions",
        )
        parser.add_argument(
            "--python-blender",
            type=int,
            default=0,
            help="Use the pythonb lending code (should be identical to C++, but may have performance differences)",
        )
        parser.add_argument(
            "--stitch-rotate-degrees",
            dest="stitch_rotate_degrees",
            type=float,
            default=None,
            help="Optional rotation (degrees) applied after stitching, about image center; keeps same dimensions.",
        )
        parser.add_argument(
            "--max-control-points",
            type=int,
            default=240,
            help="Maximum number of control points used to calculate the homography matrices",
        )
        parser.add_argument(
            "--track-ids",
            type=str,
            default=None,
            help="Comma-separated list of tracking IDs to track specifically (when online)",
        )
        #
        # Progress Bar
        #
        parser.add_argument(
            "--no-progress-bar",
            action="store_true",
            help="Don't use the progress bar",
            default=_get_disable_progress_bar(),
        )
        parser.add_argument(
            "--curses-progress",
            "--curses",
            action="store_true",
            help="Disable curses-based progress UI (use legacy printing)",
        )
        parser.add_argument(
            "--progress-bar-lines",
            type=int,
            default=11,
            help="Number of logging lines in the progrsss bar",
        )
        parser.add_argument(
            "--print-interval",
            type=int,
            default=20,
            help="How many iterations between log progress printing",
        )
        parser.add_argument(
            "--output-video-bit-rate",
            "--output_video_bit_rate",
            dest="output_video_bit_rate",
            type=int,
            default=None,
            help="Output video bit-rate",
        )

        # Jersey framework toggles (Koshkina trunk) for reuse across CLIs
        parser.add_argument(
            "--detect-jersey-numbers", action="store_true", help="Detect individual jersey numbers"
        )
        parser.add_argument(
            "--jersey-roi-mode",
            type=str,
            choices=["bbox", "pose", "sam"],
            default=None,
            help="ROI mode for jersey trunk: bbox|pose|sam",
        )
        parser.add_argument(
            "--jersey-str-backend",
            type=str,
            choices=["mmocr", "parseq"],
            default=None,
            help="STR backend: mmocr (default) or parseq",
        )
        parser.add_argument(
            "--jersey-parseq-weights", type=str, default=None, help="PARSeq weights path"
        )
        parser.add_argument(
            "--jersey-parseq-device", type=str, default=None, help="PARSeq device (e.g., cuda)"
        )
        parser.add_argument(
            "--jersey-legibility-enabled", action="store_true", help="Enable legibility filter"
        )
        parser.add_argument(
            "--jersey-legibility-weights", type=str, default=None, help="Legibility weights path"
        )
        parser.add_argument(
            "--jersey-legibility-threshold",
            type=float,
            default=None,
            help="Legibility score threshold",
        )
        parser.add_argument(
            "--jersey-reid-enabled", action="store_true", help="Enable ReID outlier removal"
        )
        parser.add_argument(
            "--jersey-reid-backend",
            type=str,
            choices=["resnet", "centroid"],
            default=None,
            help="ReID backend: resnet (default) or centroid",
        )
        parser.add_argument(
            "--jersey-reid-backbone",
            type=str,
            choices=["resnet18", "resnet34"],
            default=None,
            help="ReID resnet backbone",
        )
        parser.add_argument(
            "--jersey-reid-threshold", type=float, default=None, help="ReID Mahalanobis threshold"
        )
        parser.add_argument(
            "--jersey-centroid-reid-path",
            type=str,
            default=None,
            help="Path to centroid-reid repo/model",
        )
        parser.add_argument(
            "--jersey-centroid-reid-device", type=str, default=None, help="Device for centroid-reid"
        )
        parser.add_argument(
            "--jersey-sam-enabled", action="store_true", help="Enable SAM ROI refinement"
        )
        parser.add_argument(
            "--jersey-sam-checkpoint", type=str, default=None, help="Path to SAM checkpoint"
        )
        parser.add_argument(
            "--jersey-sam-model-type", type=str, default=None, help="SAM model type (e.g., vit_b)"
        )
        parser.add_argument("--jersey-sam-device", type=str, default=None, help="SAM device")

        #
        # Camera braking / stop-dampening controls
        #
        braking = parser.add_argument_group(
            "camera_braking",
            "Camera movement braking and stop dampening controls",
        )
        braking.add_argument(
            "--stop-on-dir-change-delay",
            default=10,
            type=int,
            help="Frames to brake to a stop on direction change (camera tracking)",
        )
        braking.add_argument(
            "--cancel-stop-on-opposite-dir",
            default=1,
            type=int,
            help="Cancel braking when inputs flip opposite (0/1)",
        )
        braking.add_argument(
            "--stop-cancel-hysteresis-frames",
            default=2,
            type=int,
            help="Consecutive opposite-direction frames required to cancel braking",
        )
        braking.add_argument(
            "--stop-delay-cooldown-frames",
            default=2,
            type=int,
            help="Cooldown frames after stop-delay finishes/cancels before another can start",
        )
        # Breakaway quick-stop knobs via CLI
        braking.add_argument(
            "--overshoot-stop-delay-count",
            default=6,
            type=int,
            help="When overshooting breakaway, brake to stop over N frames",
        )
        braking.add_argument(
            "--post-nonstop-stop-delay-count",
            default=6,
            type=int,
            help="After nonstop ends, brake to stop over N frames",
        )
        braking.add_argument(
            "--time-to-dest-speed-limit-frames",
            default=10,
            type=int,
            help="Minimum frames to reach destination along an axis when speeding up (0 disables)",
        )

        # Generic YAML overrides: --config-override rink.camera.foo.bar=VALUE (repeatable)
        overrides = parser.add_argument_group(
            "config_overrides",
            "Override any YAML config key with --config-override key=value",
        )
        overrides.add_argument(
            "--config-override",
            dest="config_overrides",
            action="append",
            default=[],
            help="Override a YAML key path (dot.notation) with a value (repeatable)",
        )

        #
        # UI controls
        #
        ui = parser.add_argument_group(
            "ui",
            "Runtime UI controls",
        )
        ui.add_argument(
            "--camera-ui",
            default=0,
            type=int,
            help="Enable runtime camera braking UI (OpenCV trackbars)",
        )
        return parser

    def parse(self, args=""):
        if args == "":
            opt = self._parser.parse_args()
        else:
            opt = self._parser.parse_args(args)
        return self.init(opt)

    # TODO: How can this be generalized with the nesting in the yaml?
    CONFIG_TO_ARGS = [
        # "model.tracker.pre_hm": "pre_hm",
        "model.tracker",
        "debug",
    ]
    ARG_TO_CONFIG_MAP: Mapping[str, Union[str, Sequence[str]]] = OrderedDict(
        [
            ("aspen_threaded", ["aspen.pipeline.threaded", "aspen.threaded_trunks"]),
            ("aspen_thread_queue_size", "aspen.pipeline.queue_size"),
            ("aspen_max_concurrent", "aspen.pipeline.max_concurrent"),
            ("aspen_thread_cuda_streams", "aspen.pipeline.cuda_streams"),
            ("aspen_thread_graph", "aspen.pipeline.graph"),
            ("display_plugin_profile", "aspen.pipeline.display_plugin_profile"),
            ("display_aspen_graph", "aspen.pipeline.display_graph"),
            ("aspen_stitching", "aspen.stitching.enabled"),
            ("blend_mode", "aspen.stitching.blend_mode"),
            ("max_blend_levels", "aspen.stitching.max_blend_levels"),
            ("stitch_auto_adjust_exposure", "aspen.stitching.auto_adjust_exposure"),
            ("python_blender", "aspen.stitching.python_blender"),
            ("no_minimize_blend", "aspen.stitching.minimize_blend"),
            ("minimize_blend", "aspen.stitching.minimize_blend"),
            ("no_cuda_streams", "aspen.stitching.no_cuda_streams"),
            ("stitch_rotate_degrees", "aspen.stitching.post_stitch_rotate_degrees"),
            ("fp16_stitch", "aspen.stitching.dtype"),
            ("stitch_pto_project_file", "aspen.stitching.pto_project_file"),
            (
                "skip_final_video_save",
                [
                    "aspen.video_out.skip_final_save",
                    "aspen.plugins.video_out.params.skip_final_save",
                ],
            ),
            ("video_encoder_backend", "aspen.video_out.encoder_backend"),
            ("output_file", "aspen.plugins.video_out.params.output_video_path"),
            ("save_frame_dir", "aspen.plugins.video_out.params.save_frame_dir"),
            ("crop_play_box", "aspen.apply_camera.crop_play_box"),
            ("no_crop", "aspen.apply_camera.crop_output_image"),
            ("end_zones", "aspen.apply_camera.end_zones"),
            ("show_image", "aspen.video_out.show_image"),
            ("show_scaled", "aspen.video_out.show_scaled"),
            (
                "output_width",
                [
                    "aspen.video_out.output_width",
                    "aspen.plugins.video_out.params.output_width",
                ],
            ),
            ("scoreboard_scale", "rink.scoreboard.scoreboard_scale"),
            ("output_video_bit_rate", "aspen.video_out.bit_rate"),
            (
                "checkerboard_input",
                ["debug.rgb_stats_check.enable", "aspen.stitching.capture_rgb_stats"],
            ),
            ("debug_play_tracker", "plot.debug_play_tracker"),
            ("plot_moving_boxes", "plot.plot_moving_boxes"),
            ("plot_trajectories", "plot.plot_trajectories"),
            ("plot_jersey_numbers", "plot.plot_jersey_numbers"),
            ("plot_actions", "plot.plot_actions"),
            ("plot_pose", "plot.plot_pose"),
            ("plot_ice_mask", "plot.plot_ice_mask"),
            ("plot_all_detections", "plot.plot_all_detections"),
            (
                "plot_tracking",
                ["plot.plot_individual_player_tracking", "plot.plot_boundaries"],
            ),
            ("no_plot_tracking_circles", "plot.plot_tracking_circles"),
            ("debug", "plot.debug_play_tracker"),
        ]
    )
    ARG_VALUE_MAP: Mapping[str, Union[Mapping[Any, Any], Callable[[Any], Any]]] = {
        "aspen_threaded": bool,
        "aspen_thread_queue_size": _coerce_aspen_queue_size,
        "aspen_max_concurrent": _coerce_aspen_max_concurrent,
        "aspen_thread_cuda_streams": bool,
        "aspen_thread_graph": bool,
        "display_plugin_profile": bool,
        "display_aspen_graph": bool,
        "aspen_stitching": bool,
        "stitch_auto_adjust_exposure": bool,
        "python_blender": bool,
        "no_minimize_blend": {True: False},
        "minimize_blend": bool,
        "no_cuda_streams": bool,
        "fp16_stitch": {True: "float16"},
        "skip_final_video_save": {True: True},
        "checkerboard_input": {True: True},
        "crop_play_box": bool,
        "no_crop": {True: False},
        "end_zones": {True: True},
        "show_image": bool,
        "debug_play_tracker": {True: True},
        "scoreboard_scale": float,
        "plot_moving_boxes": {True: True},
        "plot_trajectories": {True: True},
        "plot_jersey_numbers": {True: True},
        "plot_actions": {True: True},
        "plot_pose": {True: True},
        "plot_ice_mask": {True: True},
        "plot_tracking": {True: True},
        "no_plot_tracking_circles": {True: False},
        "debug": _debug_to_play_tracker,
    }
    ARG_SETDEFAULT = {"output_file", "save_frame_dir"}

    @staticmethod
    def apply_arg_config_overrides(
        config: Dict[str, Any],
        args: Any,
        arg_to_config: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        value_map: Optional[Mapping[str, Union[Mapping[Any, Any], Callable[[Any], Any]]]] = None,
        setdefault_args: Optional[Sequence[str]] = None,
    ) -> bool:
        """Apply CLI-ish overrides to a config dict using dot-path mappings."""
        if not isinstance(config, dict) or args is None:
            return False
        arg_to_config = arg_to_config or hm_opts.ARG_TO_CONFIG_MAP
        value_map = value_map or hm_opts.ARG_VALUE_MAP
        setdefault_set = set(setdefault_args or hm_opts.ARG_SETDEFAULT)
        changed = False
        for arg_name, cfg_paths in arg_to_config.items():
            raw_value = _get_arg_value(args, arg_name)
            if raw_value is _MISSING_ARG or raw_value is None:
                continue
            mapper = value_map.get(arg_name)
            mapped_value = raw_value
            if mapper is not None:
                try:
                    if isinstance(mapper, MappingABC):
                        if raw_value not in mapper:
                            continue
                        mapped_value = mapper[raw_value]
                    else:
                        mapped_value = mapper(raw_value)
                    if mapped_value is _SKIP_CONFIG_VALUE:
                        continue
                except Exception:
                    logger.warning("Invalid config override for %s: %r", arg_name, raw_value)
                    continue
            if isinstance(cfg_paths, str):
                paths = [cfg_paths]
            else:
                paths = list(cfg_paths)
            for path in paths:
                if path.startswith("aspen.plugins."):
                    # Avoid implicitly creating incomplete Aspen plugin stubs via CLI overrides.
                    # If the selected Aspen config doesn't declare a plugin, setting
                    # aspen.plugins.<name>.* here would create a dict without a `class`,
                    # which later fails AspenNet graph construction.
                    parts = path.split(".")
                    if len(parts) >= 3:
                        plugin_name = parts[2]
                        aspen_cfg = config.get("aspen")
                        plugins_cfg = (
                            aspen_cfg.get("plugins") if isinstance(aspen_cfg, dict) else None
                        )
                        if not isinstance(plugins_cfg, dict) or plugin_name not in plugins_cfg:
                            continue
                if arg_name in setdefault_set and _has_nested_key(config, path):
                    continue
                set_nested_value(config, path, mapped_value)
                changed = True
        return changed

    @staticmethod
    def apply_config_overrides(config: Dict[str, Any], overrides: Optional[Sequence[str]]) -> bool:
        """Apply generic dot-path overrides (``key=value``) to a config dict."""
        if not isinstance(config, dict) or not overrides:
            return False

        changed = False
        for ov in overrides:
            if not isinstance(ov, str) or "=" not in ov:
                continue
            key, val = ov.split("=", 1)
            sval = val.strip()
            lval = sval.lower()
            if lval in ("null", "none"):
                pval: Any = None
            elif lval in ("true", "false"):
                pval = lval == "true"
            else:
                try:
                    if "." in sval:
                        pval = float(sval)
                    else:
                        pval = int(sval)
                except Exception:
                    pval = sval
            try:
                set_nested_value(config, key.strip(), pval)
                changed = True
            except Exception:
                pass
        return changed

    @staticmethod
    def init(opt, parser: Optional[argparse.ArgumentParser] = None):
        # Normalize some conflicting arguments
        if opt.serial:
            opt.cache_size = 0
            opt.no_async_dataset = True

        if opt.show_scaled:
            opt.show_image = True

        # Resolve "auto" decoder selection to a concrete backend.
        # Prefer GPU decode (pynvcodec) when CUDA + PyNvVideoCodec are available.
        try:
            method = getattr(opt, "video_stream_decode_method", None)
            key = method.strip().lower() if isinstance(method, str) else ""
            if key in ("", "auto", "cuda"):
                chosen = "cv2"
                cuda_ok = False
                try:
                    import torch

                    cuda_ok = bool(torch.cuda.is_available())
                except Exception:
                    cuda_ok = False
                if cuda_ok:
                    try:
                        import importlib.util

                        if importlib.util.find_spec("PyNvVideoCodec") is not None:
                            chosen = "pynvcodec"
                    except Exception:
                        chosen = "cv2"
                opt.video_stream_decode_method = chosen
        except Exception:
            pass

        for key in hm_opts.CONFIG_TO_ARGS:
            nested_item = get_nested_value(getattr(opt, "game_config", {}), key, None)
            if nested_item is None:
                continue
            if isinstance(nested_item, dict):
                for k, v in nested_item.items():
                    if hasattr(opt, k):
                        current_val = getattr(opt, k)
                        if current_val is None or (
                            parser is not None and current_val == parser.get_default(k)
                        ):
                            print(f"Setting attribute {k} to {v}")
                            setattr(opt, k, v)

        # YAML-derived defaults for stitch_rotate_degrees (CLI wins).
        # Preferred source is consolidated game_config when available; otherwise
        # fall back to the per-game private config under $HOME/Videos/<game-id>/config.yaml.
        try:
            if getattr(opt, "stitch_rotate_degrees", None) is None:
                val = None
                game_cfg = getattr(opt, "game_config", None)
                if isinstance(game_cfg, dict):
                    val = get_nested_value(game_cfg, "game.stitching.stitch-rotate-degrees", None)
                    if val is None:
                        val = get_nested_value(
                            game_cfg, "game.stitching.stitch_rotate_degrees", None
                        )
                if val is None and getattr(opt, "game_id", None):
                    cfg_priv = get_game_config_private(game_id=opt.game_id)
                    val = get_nested_value(cfg_priv, "game.stitching.stitch-rotate-degrees", None)
                    if val is None:
                        val = get_nested_value(
                            cfg_priv, "game.stitching.stitch_rotate_degrees", None
                        )
                if val is not None:
                    try:
                        opt.stitch_rotate_degrees = float(val)
                    except Exception:
                        pass
        except Exception:
            # Non-fatal if config missing or malformed
            pass

        # Map select CLI camera options into YAML-style config (if not already present)
        # This lets downstream code read from args.game_config['rink']['camera']
        game_cfg = getattr(opt, "game_config", None)
        if isinstance(game_cfg, dict):
            # Helper to know if a CLI option was explicitly provided (not just default)
            def _cli_spec(name: str) -> bool:
                try:
                    return parser is not None and getattr(opt, name) != parser.get_default(name)
                except Exception:
                    return False

            # cam_ignore_largest: prefer explicit CLI flag, otherwise YAML rink.tracking setting.
            try:
                cfg_val = get_nested_value(game_cfg, "rink.tracking.cam_ignore_largest", None)
                if cfg_val is not None and not _cli_spec("cam_ignore_largest"):
                    opt.cam_ignore_largest = bool(cfg_val)
            except Exception:
                pass

            # Camera controller/model/window: only override YAML when explicitly specified on CLI.
            try:
                if hasattr(opt, "camera_controller") and (
                    _cli_spec("camera_controller")
                    or get_nested_value(game_cfg, "rink.camera.controller", None) is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.controller",
                        str(getattr(opt, "camera_controller")),
                    )
            except Exception:
                pass
            try:
                if hasattr(opt, "camera_model") and (
                    _cli_spec("camera_model")
                    or get_nested_value(game_cfg, "rink.camera.camera_model", None) is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.camera_model",
                        getattr(opt, "camera_model"),
                    )
            except Exception:
                pass
            try:
                if hasattr(opt, "camera_window") and (
                    _cli_spec("camera_window")
                    or get_nested_value(game_cfg, "rink.camera.camera_window", None) is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.camera_window",
                        int(getattr(opt, "camera_window")),
                    )
            except Exception:
                pass

            # stop_on_dir_change_delay
            try:
                if (
                    _cli_spec("stop_on_dir_change_delay")
                    or get_nested_value(game_cfg, "rink.camera.stop_on_dir_change_delay", None)
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.stop_on_dir_change_delay",
                        int(opt.stop_on_dir_change_delay),
                    )
            except Exception:
                pass
            # cancel_stop_on_opposite_dir (store as bool in YAML)
            try:
                if (
                    _cli_spec("cancel_stop_on_opposite_dir")
                    or get_nested_value(game_cfg, "rink.camera.cancel_stop_on_opposite_dir", None)
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.cancel_stop_on_opposite_dir",
                        bool(int(opt.cancel_stop_on_opposite_dir)),
                    )
            except Exception:
                pass
            # cancel hysteresis frames
            try:
                if (
                    _cli_spec("stop_cancel_hysteresis_frames")
                    or get_nested_value(game_cfg, "rink.camera.stop_cancel_hysteresis_frames", None)
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.stop_cancel_hysteresis_frames",
                        int(opt.stop_cancel_hysteresis_frames),
                    )
            except Exception:
                pass
            # stop delay cooldown frames
            try:
                if (
                    _cli_spec("stop_delay_cooldown_frames")
                    or get_nested_value(game_cfg, "rink.camera.stop_delay_cooldown_frames", None)
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.stop_delay_cooldown_frames",
                        int(opt.stop_delay_cooldown_frames),
                    )
            except Exception:
                pass
            # time to dest speed limit frames
            try:
                if (
                    _cli_spec("time_to_dest_speed_limit_frames")
                    or get_nested_value(
                        game_cfg, "rink.camera.time_to_dest_speed_limit_frames", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.time_to_dest_speed_limit_frames",
                        int(opt.time_to_dest_speed_limit_frames),
                    )
            except Exception:
                pass
            # resizing stop_on_dir_change_delay
            try:
                if (
                    _cli_spec("resizing_stop_on_dir_change_delay")
                    or get_nested_value(
                        game_cfg, "rink.camera.resizing_stop_on_dir_change_delay", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.resizing_stop_on_dir_change_delay",
                        int(opt.resizing_stop_on_dir_change_delay),
                    )
            except Exception:
                pass
            # resizing cancel_stop_on_opposite_dir (store as bool in YAML)
            try:
                if (
                    _cli_spec("resizing_cancel_stop_on_opposite_dir")
                    or get_nested_value(
                        game_cfg, "rink.camera.resizing_cancel_stop_on_opposite_dir", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.resizing_cancel_stop_on_opposite_dir",
                        bool(int(opt.resizing_cancel_stop_on_opposite_dir)),
                    )
            except Exception:
                pass
            # resizing cancel hysteresis frames
            try:
                if (
                    _cli_spec("resizing_stop_cancel_hysteresis_frames")
                    or get_nested_value(
                        game_cfg, "rink.camera.resizing_stop_cancel_hysteresis_frames", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.resizing_stop_cancel_hysteresis_frames",
                        int(opt.resizing_stop_cancel_hysteresis_frames),
                    )
            except Exception:
                pass
            # resizing stop delay cooldown frames
            try:
                if (
                    _cli_spec("resizing_stop_delay_cooldown_frames")
                    or get_nested_value(
                        game_cfg, "rink.camera.resizing_stop_delay_cooldown_frames", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.resizing_stop_delay_cooldown_frames",
                        int(opt.resizing_stop_delay_cooldown_frames),
                    )
            except Exception:
                pass
            # resizing time to dest speed limit frames
            try:
                if (
                    _cli_spec("resizing_time_to_dest_speed_limit_frames")
                    or get_nested_value(
                        game_cfg,
                        "rink.camera.resizing_time_to_dest_speed_limit_frames",
                        None,
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.resizing_time_to_dest_speed_limit_frames",
                        int(opt.resizing_time_to_dest_speed_limit_frames),
                    )
            except Exception:
                pass
            # Breakaway: overshoot/post-nonstop delays
            try:
                if _cli_spec("overshoot_stop_delay_count") or (
                    get_nested_value(
                        game_cfg, "rink.camera.breakaway_detection.overshoot_stop_delay_count", None
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.breakaway_detection.overshoot_stop_delay_count",
                        int(opt.overshoot_stop_delay_count),
                    )
            except Exception:
                pass
            try:
                if _cli_spec("post_nonstop_stop_delay_count") or (
                    get_nested_value(
                        game_cfg,
                        "rink.camera.breakaway_detection.post_nonstop_stop_delay_count",
                        None,
                    )
                    is None
                ):
                    set_nested_value(
                        game_cfg,
                        "rink.camera.breakaway_detection.post_nonstop_stop_delay_count",
                        int(opt.post_nonstop_stop_delay_count),
                    )
            except Exception:
                pass

            # Apply generic overrides: key=value, with simple type inference
            for ov in opt.config_overrides or []:
                if not isinstance(ov, str) or "=" not in ov:
                    continue
                key, val = ov.split("=", 1)
                sval = val.strip()
                # Try to infer type: null/None, bool, int, float; else keep string
                lval = sval.lower()
                if lval in ("null", "none"):
                    pval: Any = None
                elif lval in ("true", "false"):
                    pval = lval == "true"
                else:
                    try:
                        if "." in sval:
                            pval = float(sval)
                        else:
                            pval = int(sval)
                    except Exception:
                        pval = sval
                try:
                    set_nested_value(game_cfg, key.strip(), pval)
                except Exception:
                    pass
        else:
            # If there's no game_config yet, create one to hold overrides
            if getattr(opt, "config_overrides", []):
                opt.game_config = {}
                for ov in opt.config_overrides or []:
                    if not isinstance(ov, str) or "=" not in ov:
                        continue
                    key, val = ov.split("=", 1)
                    sval = val.strip()
                    lval = sval.lower()
                    if lval in ("null", "none"):
                        pval = None
                    elif lval in ("true", "false"):
                        pval = lval == "true"
                    else:
                        try:
                            if "." in sval:
                                pval = float(sval)
                            else:
                                pval = int(sval)
                        except Exception:
                            pval = sval
                    try:
                        set_nested_value(opt.game_config, key.strip(), pval)
                    except Exception:
                        pass

        return opt


def preferred_arg(preferred_arg: Any, backup_arg: Any):
    """Return ``preferred_arg`` if not ``None``, otherwise ``backup_arg``."""
    if preferred_arg is not None:
        return preferred_arg
    return backup_arg


def add_remaining_autogenerated(parser: argparse.ArgumentParser):
    # parser.add_argument("--cameramake", default=None, type=str, help="Cameramake")
    # parser.add_argument("--cameramodel", default=None, type=str, help="Cameramodel")
    parser.add_argument("--cameraoutput-fps", default=None, type=str, help="Cameraoutput Fps")
    parser.add_argument(
        "--cameramount",
        default=[{"offset_80": None, "resolution": None}, {"offset_90": None, "resolution": None}],
        type=str,
        help="Cameramount",
    )
    parser.add_argument("--rinkname", default=None, type=str, help="Rinkname")
    parser.add_argument("--rinklocation-city", default=None, type=str, help="Rinklocation City")
    parser.add_argument("--rinklocation-state", default=None, type=str, help="Rinklocation State")
    parser.add_argument(
        "--rinklocation-country", default=None, type=str, help="Rinklocation Country"
    )
    parser.add_argument(
        "--rinkdimensions-length", default=None, type=str, help="Rinkdimensions Length"
    )
    parser.add_argument(
        "--rinkdimensions-width", default=None, type=str, help="Rinkdimensions Width"
    )
    parser.add_argument(
        "--rinkdimensions-corner-radius",
        default=None,
        type=str,
        help="Rinkdimensions Corner Radius",
    )
    parser.add_argument(
        "--rinkseating-capacity", default=None, type=str, help="Rinkseating Capacity"
    )
    parser.add_argument(
        "--rinkteams-home-team-name", default=None, type=str, help="Rinkteams Home Team Name"
    )
    parser.add_argument(
        "--rinkteams-home-team-colors", default=None, type=str, help="Rinkteams Home Team Colors"
    )
    parser.add_argument(
        "--rinkfacilities-locker-rooms", default=None, type=str, help="Rinkfacilities Locker Rooms"
    )
    parser.add_argument(
        "--rinkfacilities-concession-stands",
        default=None,
        type=str,
        help="Rinkfacilities Concession Stands",
    )
    parser.add_argument(
        "--rinkfacilities-restrooms", default=None, type=str, help="Rinkfacilities Restrooms"
    )
    parser.add_argument(
        "--rinkparking-capacity", default=None, type=str, help="Rinkparking Capacity"
    )
    parser.add_argument("--rinkparking-price", default=None, type=str, help="Rinkparking Price")
    parser.add_argument(
        "--rinkscoreboard-perspective-polygon",
        default=None,
        type=str,
        help="Rinkscoreboard Perspective Polygon",
    )
    parser.add_argument(
        "--rinkscoreboard-projected-height",
        default="%20",
        type=str,
        help="Rinkscoreboard Projected Height",
    )
    parser.add_argument(
        "--rinkscoreboard-projected-width",
        default="%10",
        type=str,
        help="Rinkscoreboard Projected Width",
    )
    parser.add_argument(
        "--rinkend-zones-left-start", default=None, type=str, help="Rinkend Zones Left Start"
    )
    parser.add_argument(
        "--rinkend-zones-left-stop", default=None, type=str, help="Rinkend Zones Left Stop"
    )
    parser.add_argument(
        "--rinkend-zones-right-start", default=None, type=str, help="Rinkend Zones Right Start"
    )
    parser.add_argument(
        "--rinkend-zones-right-stop", default=None, type=str, help="Rinkend Zones Right Stop"
    )
    parser.add_argument(
        "--rinktracking-cam-ignore-largest",
        default=True,
        type=int,
        help="Rinktracking Cam Ignore Largest",
    )
    parser.add_argument(
        "--rinkcamera-fixed-edge-scaling-factor",
        default=0.8,
        type=str,
        help="Rinkcamera Fixed Edge Scaling Factor",
    )
    parser.add_argument(
        "--rinkcamera-fixed-edge-rotation-angle",
        default=30,
        type=str,
        help="Rinkcamera Fixed Edge Rotation Angle",
    )
    parser.add_argument(
        "--rinkcamera-image-channel-adjustment",
        default=None,
        type=str,
        help="Rinkcamera Image Channel Adjustment",
    )
    parser.add_argument(
        "--rinkcamera-follower-box-scale-width",
        default=1.25,
        type=str,
        help="Rinkcamera Follower Box Scale Width",
    )
    parser.add_argument(
        "--rinkcamera-follower-box-scale-height",
        default=1.25,
        type=str,
        help="Rinkcamera Follower Box Scale Height",
    )
    parser.add_argument(
        "--rinkcamera-sticky-size-ratio-to-frame-width",
        default=10.0,
        type=str,
        help="Rinkcamera Sticky Size Ratio To Frame Width",
    )
    parser.add_argument(
        "--rinkcamera-sticky-translation-gaussian-mult",
        default=5.0,
        type=str,
        help="Rinkcamera Sticky Translation Gaussian Mult",
    )
    parser.add_argument(
        "--rinkcamera-unsticky-translation-size-ratio",
        default=0.75,
        type=str,
        help="Rinkcamera Unsticky Translation Size Ratio",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-min-considered-group-velocity",
        default=3.0,
        type=str,
        help="Rinkcamera Breakaway Detection Min Considered Group Velocity",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-group-ratio-threshold",
        default=0.5,
        type=str,
        help="Rinkcamera Breakaway Detection Group Ratio Threshold",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-group-velocity-speed-ratio",
        default=0.3,
        type=str,
        help="Rinkcamera Breakaway Detection Group Velocity Speed Ratio",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-scale-speed-constraints",
        default=2.0,
        type=str,
        help="Rinkcamera Breakaway Detection Scale Speed Constraints",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-nonstop-delay-count",
        default=2,
        type=str,
        help="Rinkcamera Breakaway Detection Nonstop Delay Count",
    )
    parser.add_argument(
        "--rinkcamera-breakaway-detection-overshoot-scale-speed-ratio",
        default=0.7,
        type=str,
        help="Rinkcamera Breakaway Detection Overshoot Scale Speed Ratio",
    )
    parser.add_argument("--gamename", default=None, type=str, help="Gamename")
    parser.add_argument("--gamerink", default=None, type=str, help="Gamerink")
    parser.add_argument("--gamehome", default=None, type=str, help="Gamehome")
    parser.add_argument("--gameaway", default=None, type=str, help="Gameaway")
    parser.add_argument(
        "--gamestitching-frame-offsets-left",
        default=None,
        type=str,
        help="Gamestitching Frame Offsets Left",
    )
    parser.add_argument(
        "--gamestitching-frame-offsets-right",
        default=None,
        type=str,
        help="Gamestitching Frame Offsets Right",
    )
    parser.add_argument(
        "--gamestitching-control-points-m-kpts0",
        default=None,
        type=str,
        help="Gamestitching Control Points M Kpts0",
    )
    parser.add_argument(
        "--gamestitching-control-points-m-kpts1",
        default=None,
        type=str,
        help="Gamestitching Control Points M Kpts1",
    )
    parser.add_argument(
        "--gamestitching-offsets", default=None, type=str, help="Gamestitching Offsets"
    )
    parser.add_argument("--gameclip-box", default=None, type=str, help="Gameclip Box")
    parser.add_argument(
        "--gameboundaries-upper", default=None, type=str, help="Gameboundaries Upper"
    )
    parser.add_argument(
        "--gameboundaries-upper-tune-position",
        default=None,
        type=str,
        help="Gameboundaries Upper Tune Position",
    )
    parser.add_argument(
        "--gameboundaries-lower", default=None, type=str, help="Gameboundaries Lower"
    )
    parser.add_argument(
        "--gameboundaries-lower-tune-position",
        default=None,
        type=str,
        help="Gameboundaries Lower Tune Position",
    )
    parser.add_argument(
        "--gameboundaries-scale-width", default=None, type=str, help="Gameboundaries Scale Width"
    )
    parser.add_argument(
        "--gameboundaries-scale-height", default=None, type=str, help="Gameboundaries Scale Height"
    )
    parser.add_argument(
        "--modelend-to-end-config",
        default="config/models/hm2/hm_end_to_end.py",
        type=str,
        help="Modelend To End Config",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint",
        default="pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch_with_detector_prefix.pth",
        type=str,
        help="Modelend To End Checkpoint",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint-local",
        default="pretrained/mmdetection/yolox_s_8x8_300e_coco_80e_ch_with_detector_prefix.pth",
        type=str,
        help="Modelend To End Checkpoint Local",
    )
    parser.add_argument(
        "--modelend-to-end-checkpoint-remote",
        default="https://drive.google.com/file/d/1WgJ-u2aL1Yv6VNXF5w-0DtsxCCDXAEe7/view?usp=drive_link",
        type=str,
        help="Modelend To End Checkpoint Remote",
    )
    parser.add_argument(
        "--modelpose-config",
        default="openmm/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py",
        type=str,
        help="Modelpose Config",
    )
    parser.add_argument(
        "--modelpose-checkpoint",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth",
        type=str,
        help="Modelpose Checkpoint",
    )
    parser.add_argument(
        "--modelpose-checkpoint-remote",
        default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth",
        type=str,
        help="Modelpose Checkpoint Remote",
    )
    parser.add_argument(
        "--modelice-rink-segm-config",
        default="config/models/ice_rink/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py",
        type=str,
        help="Modelice Rink Segm Config",
    )
    parser.add_argument(
        "--modelice-rink-segm-checkpoint-local",
        default="pretrained/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/ice_rink_iter_30000.pth",
        type=str,
        help="Modelice Rink Segm Checkpoint Local",
    )
    parser.add_argument(
        "--modelice-rink-segm-checkpoint",
        default="pretrained/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/ice_rink_iter_30000.pth",
        type=str,
        help="Modelice Rink Segm Checkpoint",
    )
    parser.add_argument(
        "--modelsvnh-classifier-config", default=None, type=str, help="Modelsvnh Classifier Config"
    )
    parser.add_argument(
        "--modelsvnh-classifier-checkpoint",
        default="pretrained/svhnc/model-65000.pth",
        type=str,
        help="Modelsvnh Classifier Checkpoint",
    )


# TODO: FIXME, doesn;t properly handle nested item names converted to arg names (no dash)
def generate_yaml_args_code(parser: argparse.ArgumentParser, yaml_file_path: Path) -> str:
    """
    Generates Python code to add arguments from a YAML file to an argparse parser object, including nested YAML items.

    Args:
        parser (argparse.ArgumentParser): An argparse parser object.
        yaml_file_path (Path): The path to the YAML file containing the arguments.

    Returns:
        str: A string containing the Python code to add the arguments to the parser.
    """
    # Read the YAML file
    with open(yaml_file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    code_lines = []

    def to_camel_case(snake_str: str) -> str:
        components = snake_str.split("-")
        return " ".join(x.title() for x in components)

    def process_yaml_items(prefix: str, data: Union[Dict[str, Any], Any]) -> None:
        if isinstance(data, dict):
            for key, value in data.items():
                if key.endswith("_description") or key.endswith("_type"):
                    # Skip _description and _type entries for now
                    continue

                new_prefix = f"{prefix}-{key}-" if prefix else key
                process_yaml_items(new_prefix, value)
        else:
            # Replace underscores with dashes for the argument name
            arg_name = prefix.rstrip("-").replace("_", "-")

            # Determine help description and type
            description = yaml_data.get(
                f'{prefix.rstrip("-")}_description', to_camel_case(arg_name)
            )
            value_type = yaml_data.get(f'{prefix.rstrip("-")}_type', str)

            if isinstance(data, bool):
                value_type = int  # Change boolean type to int for argparse

            # Check if the argument already exists in the parser
            if not any(arg_name == action.dest for action in parser._actions):
                # Generate the code to add the argument to the parser
                code_line = f"parser.add_argument('--{arg_name}', default={repr(data)}, type={value_type.__name__}, help='{description}')"
                code_lines.append(code_line)

    process_yaml_items("", yaml_data)
    return "\n    ".join(code_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Python code to add YAML configuration to argparse parser"
    )
    parser.add_argument("yaml_file_path", type=Path, help="Path to the YAML file")

    args = parser.parse_args()
    generated_code = generate_yaml_args_code(hm_opts.parser(), args.yaml_file_path)

    # Print the generated code
    # TODO: Generate the
    print("def add_remaining_autogenerated(parser: argparse.ArgumentParser):\n    ")
    print(generated_code)
