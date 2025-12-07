import contextlib
import traceback
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

# AspenNet graph runner
from hmlib.aspen import AspenNet
from hmlib.config import get_game_dir
from hmlib.datasets.dataframe import find_latest_dataframe_file
from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.utils import MeanTracker
from hmlib.utils.gpu import cuda_stream_scope
from hmlib.utils.image import make_channels_first
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, convert_seconds_to_hms


def run_mmtrack(
    model,
    pose_inferencer,
    config: Dict[str, Any],
    dataloader,
    postprocessor,
    progress_bar: Optional[ProgressBar] = None,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
    no_cuda_streams: bool = False,
    track_mean_mode: Optional[str] = None,
    profiler: Any = None,
):
    mean_tracker: Optional[MeanTracker] = None
    aspen_net: Optional[AspenNet] = None
    if config is None:
        config = {}
    try:
        if track_mean_mode:
            mean_tracker = MeanTracker(
                file_path="pre_detect.txt",
                mode=track_mean_mode,
            )
        cuda_stream = torch.cuda.Stream(device) if not no_cuda_streams else None
        with cuda_stream_scope(cuda_stream):
            dataloader_iterator = CachedIterator(
                iterator=iter(dataloader), cache_size=input_cache_size
            )
            # print("WARNING: Not cacheing data loader")

            #
            # Calculate some dataset stats for our progress display
            #
            batch_size = dataloader.batch_size
            fps: Optional[float] = getattr(dataloader, "fps", None)
            total_batches_in_video = len(dataloader)
            total_frames_in_video = batch_size * total_batches_in_video
            number_of_batches_processed = 0
            total_duration_str = convert_seconds_to_hms(total_frames_in_video / dataloader.fps)
            nr_tracks = 0

            if model is not None:
                model.eval()

            wraparound_timer = None
            detect_timer = None
            last_frame_id = None
            max_tracking_id = 0

            if progress_bar is not None:
                dataloader_iterator = progress_bar.set_iterator(dataloader_iterator)

                #
                # The progress table
                #
                def _table_callback(table_map: OrderedDict[Any, Any]):
                    duration_processed_in_seconds = (
                        number_of_batches_processed * batch_size / dataloader.fps
                    )
                    remaining_frames_to_process = total_frames_in_video - (
                        number_of_batches_processed * batch_size
                    )
                    remaining_seconds_to_process = remaining_frames_to_process / dataloader.fps

                    if wraparound_timer is not None:
                        processing_fps = batch_size / max(1e-5, wraparound_timer.average_time)

                        table_map["HMTrack FPS"] = "{:.2f}".format(processing_fps)
                        table_map["Dataset length"] = total_duration_str
                        table_map["Processed"] = convert_seconds_to_hms(
                            duration_processed_in_seconds
                        )
                        table_map["Remaining"] = convert_seconds_to_hms(
                            remaining_seconds_to_process
                        )
                        table_map["ETA"] = convert_seconds_to_hms(
                            remaining_frames_to_process / processing_fps
                        )
                        table_map["Track count"] = str(nr_tracks)
                        table_map["Track IDs"] = str(int(max_tracking_id))

                # Add that table-maker to the progress bar
                progress_bar.add_table_callback(_table_callback)

            game_dir = config.get("game_dir")
            if not game_dir and config.get("game_id"):
                try:
                    game_dir = get_game_dir(config.get("game_id"), assert_exists=False)
                except Exception:
                    game_dir = None
            work_dir = config.get("work_dir") or config.get("results_folder")
            tracking_data_path = config.get("tracking_data_path") or find_latest_dataframe_file(
                game_dir, "tracking"
            )
            detection_data_path = config.get("detection_data_path") or find_latest_dataframe_file(
                game_dir, "detections"
            )
            pose_data_path = config.get("pose_data_path") or find_latest_dataframe_file(
                game_dir, "pose"
            )
            action_data_path = config.get("action_data_path") or find_latest_dataframe_file(
                game_dir, "actions"
            )

            # using_precalculated_tracking = bool(tracking_data_path)
            # using_precalculated_detection = bool(detection_data_path)
            # using_precalculated_pose = bool(pose_data_path)

            # Build AspenNet if a config is provided under config['aspen']
            aspen_cfg: Optional[Dict[str, Any]] = None
            cfg_aspen = config.get("aspen")
            if isinstance(cfg_aspen, dict):
                aspen_cfg = dict(cfg_aspen)
            if aspen_cfg:
                trunks_cfg = aspen_cfg.get("plugins", {}) or {}
                initial_args = config.get("initial_args", {}) or {}

                aspen_cfg["plugins"] = trunks_cfg

                pipeline_cfg = dict(aspen_cfg.get("pipeline", {}) or {})
                pipeline_modified = bool(pipeline_cfg)
                threaded_cli = initial_args.get("aspen_threaded")
                if threaded_cli is not None:
                    threaded_bool = bool(threaded_cli)
                    pipeline_cfg["threaded"] = threaded_bool
                    aspen_cfg["threaded_trunks"] = threaded_bool
                    pipeline_modified = True
                queue_cli = initial_args.get("aspen_thread_queue_size")
                if queue_cli is not None:
                    try:
                        pipeline_cfg["queue_size"] = max(1, int(queue_cli))
                        pipeline_modified = True
                    except Exception:
                        logger.warning("Invalid Aspen queue size override: %r", queue_cli)
                stream_cli = initial_args.get("aspen_thread_cuda_streams")
                if stream_cli is not None:
                    pipeline_cfg["cuda_streams"] = bool(stream_cli)
                    pipeline_modified = True
                if pipeline_modified:
                    aspen_cfg["pipeline"] = pipeline_cfg

                # Apply camera controller CLI overrides if present
                if "camera_controller" in trunks_cfg:
                    cc = trunks_cfg["camera_controller"]
                    cc_params = cc.setdefault("params", {}) or {}
                    ctrl = initial_args.get("camera_controller")
                    if ctrl:
                        cc_params["controller"] = ctrl
                    model_path = initial_args.get("camera_model")
                    if model_path:
                        cc_params["model_path"] = model_path
                    win = initial_args.get("camera_window")
                    if win:
                        try:
                            cc_params["window"] = int(win)
                        except Exception:
                            pass
                    # If CLI overrides not provided, pull from rink.camera config
                    if not cc_params.get("controller"):
                        cam_ctrl = (config.get("rink") or {}).get("camera", {}).get("controller")
                        if cam_ctrl:
                            cc_params["controller"] = cam_ctrl
                    if not cc_params.get("model_path"):
                        cam_model = (config.get("rink") or {}).get("camera", {}).get("camera_model")
                        if cam_model:
                            cc_params["model_path"] = cam_model
                    if not cc_params.get("window"):
                        cam_win = (config.get("rink") or {}).get("camera", {}).get("camera_window")
                        if cam_win:
                            try:
                                cc_params["window"] = int(cam_win)
                            except Exception:
                                pass
                    cc["params"] = cc_params
                # Apply jersey trunk CLI overrides if present
                if "jersey_numbers" in trunks_cfg:
                    j = trunks_cfg["jersey_numbers"]
                    j_params = j.setdefault("params", {}) or {}

                    def set_if(name_cfg: str, name_arg: str, allow_false: bool = False):
                        val = config.get(name_arg)
                        if val is None:
                            return
                        if isinstance(val, bool) and not val and not allow_false:
                            # Only set booleans when True unless explicitly allowed
                            return
                        j_params[name_cfg] = val

                    # ROI / SAM
                    set_if("roi_mode", "jersey_roi_mode")
                    set_if("sam_enabled", "jersey_sam_enabled")
                    set_if("sam_checkpoint", "jersey_sam_checkpoint")
                    set_if("sam_model_type", "jersey_sam_model_type")
                    set_if("sam_device", "jersey_sam_device")
                    # STR
                    set_if("str_backend", "jersey_str_backend")
                    set_if("parseq_weights", "jersey_parseq_weights")
                    set_if("parseq_device", "jersey_parseq_device")
                    # Legibility
                    set_if("legibility_enabled", "jersey_legibility_enabled")
                    set_if("legibility_weights", "jersey_legibility_weights")
                    set_if("legibility_threshold", "jersey_legibility_threshold", allow_false=True)
                    # ReID
                    set_if("reid_enabled", "jersey_reid_enabled")
                    set_if("reid_backend", "jersey_reid_backend")
                    set_if("reid_backbone", "jersey_reid_backbone")
                    set_if("reid_threshold", "jersey_reid_threshold", allow_false=True)
                    set_if("centroid_reid_path", "jersey_centroid_reid_path")
                    set_if("centroid_reid_device", "jersey_centroid_reid_device")
                    j["params"] = j_params
                # Disable postprocess trunk when CLI requests no play tracking.
                if config.get("no_play_tracking") and "postprocess" in trunks_cfg:
                    pp = trunks_cfg["postprocess"]
                    if isinstance(pp, dict):
                        pp["enabled"] = False
                shared = dict(
                    model=model,
                    pose_inferencer=pose_inferencer,
                    postprocessor=postprocessor,
                    fp16=fp16,
                    device=device,
                    # using_precalculated_tracking=using_precalculated_tracking,
                    # using_precalculated_detection=using_precalculated_detection,
                    plot_pose=bool(config.get("plot_pose", False)),
                    # Propagate CLI flag to BoundariesPlugin -> IceRinkSegmBoundaries(draw)
                    plot_ice_mask=bool(config.get("plot_ice_mask", False)),
                    # Boundary + identity context for plugins
                    game_id=config.get("game_id"),
                    game_dir=game_dir,
                    work_dir=work_dir,
                    tracking_data_path=tracking_data_path,
                    detection_data_path=detection_data_path,
                    pose_data_path=pose_data_path,
                    action_data_path=action_data_path,
                    original_clip_box=config.get("original_clip_box"),
                    top_border_lines=config.get("top_border_lines"),
                    bottom_border_lines=config.get("bottom_border_lines"),
                    # Full game config and CLI-derived initial args for plugins
                    game_config=config.get("game_config"),
                    initial_args=config.get("initial_args"),
                    # Runtime camera braking UI toggle (OpenCV trackbars) for PlayTrackerPlugin
                    camera_ui=int(initial_args.get("camera_ui") or config.get("camera_ui") or 0),
                    # Optional stitching rotation controller (e.g., StitchDataset instance)
                    stitch_rotation_controller=config.get("stitch_rotation_controller"),
                )
                if profiler is not None:
                    shared["profiler"] = profiler
                aspen_name = aspen_cfg.get("name") or config.get("game_id") or "aspen"
                aspen_net = AspenNet(aspen_name, aspen_cfg, shared=shared)
                aspen_net = aspen_net.to(device)
            # Optional torch profiler context spanning the run
            prof_ctx = profiler if getattr(profiler, "enabled", False) else contextlib.nullcontext()
            with prof_ctx:
                for cur_iter, dataset_results in enumerate(dataloader_iterator):
                    data_item = dataset_results.pop("pano")
                    original_images = data_item.pop("original_images")
                    info_imgs = data_item["img_info"]
                    ids = data_item["ids"]

                    if fps:
                        data_item["fps"] = fps
                    with torch.no_grad():
                        frame_id = info_imgs["frame_id"]

                    batch_size = original_images.shape[0]

                    if last_frame_id is None:
                        last_frame_id = int(frame_id)
                    else:
                        assert int(frame_id) == last_frame_id + batch_size
                        last_frame_id = int(frame_id)

                    batch_size = original_images.shape[0]

                    if detect_timer is None:
                        detect_timer = Timer()

                    if aspen_net is not None:
                        # Execute the configured DAG
                        # Prepare per-iteration context
                        iter_context: Dict[str, Any] = dict(
                            original_images=make_channels_first(original_images),
                            data=data_item,
                            ids=ids,
                            info_imgs=info_imgs,
                            frame_id=int(frame_id),
                            device=device,
                            cuda_stream=cuda_stream,
                            detect_timer=detect_timer,
                            mean_tracker=mean_tracker,
                        )
                        # Merge shared into context for plugins convenience
                        iter_context.update(aspen_net.shared)
                        if dataset_results:
                            iter_context.setdefault("data", {})["dataset_results"] = dataset_results

                        if getattr(profiler, "enabled", False):
                            with profiler.rf("aspen.forward"):
                                out_context = aspen_net(iter_context)
                        else:
                            out_context = aspen_net(iter_context)

                        # Async AspenNet returns None from forward()
                        if out_context is not None:
                            # Update stats for progress bar
                            nr_tracks = int(out_context.get("nr_tracks", 0))
                            max_tracking_id = out_context.get("max_tracking_id", 0)
                            if not isinstance(max_tracking_id, (int, float)):
                                try:
                                    max_tracking_id = int(max_tracking_id)
                                except Exception:
                                    max_tracking_id = 0
                    else:
                        # Legacy MMTracking path has been removed. An Aspen config is required.
                        raise RuntimeError(
                            "AspenNet config is required. Legacy non-Aspen pipeline has been removed."
                        )

                    if detect_timer is not None and cur_iter % 50 == 0:
                        # print(
                        logger.info(
                            "AspenNet iteration, frame {} ({:.2f} fps)".format(
                                frame_id,
                                batch_size * 1.0 / max(1e-5, detect_timer.average_time),
                            )
                        )

                    if cur_iter % 200 == 0:
                        if wraparound_timer is not None:
                            wraparound_timer.toc()
                            logger.info(
                                "Full wraparound time, frame {} ({:.2f} fps)".format(
                                    frame_id,
                                    batch_size * 1.0 / max(1e-5, wraparound_timer.average_time),
                                )
                            )
                        wraparound_timer = Timer()
                    elif wraparound_timer is None:
                        wraparound_timer = Timer()
                    else:
                        wraparound_timer.toc()

                    number_of_batches_processed += 1
                    # Per-iteration profiler step for per-iter export if enabled
                    if getattr(profiler, "enabled", False):
                        profiler.step()
                    wraparound_timer.tic()
    except StopIteration:
        print("run_mmtrack reached end of dataset")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if aspen_net is not None:
            try:
                aspen_net.finalize()
            except Exception:
                logger.exception("AspenNet finalize failed")
        if mean_tracker is not None:
            mean_tracker.close()
