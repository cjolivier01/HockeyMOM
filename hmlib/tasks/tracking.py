from collections import OrderedDict
from typing import Any, Dict, Optional
import contextlib

import torch
import yaml

import hmlib.models.end_to_end  # Registers the model
import hmlib.tracking_utils.segm_boundaries

# AspenNet graph runner
from hmlib.aspen import AspenNet
from hmlib.log import logger
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
from hmlib.tracking_utils.timer import Timer
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils import MeanTracker
from hmlib.utils.gpu import cuda_stream_scope
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, convert_seconds_to_hms


def run_mmtrack(
    model,
    pose_inferencer,
    config: Dict[str, Any],
    dataloader,
    postprocessor,
    progress_bar: Optional[ProgressBar] = None,
    tracking_dataframe: TrackingDataFrame = None,
    detection_dataframe: DetectionDataFrame = None,
    pose_dataframe: PoseDataFrame = None,
    action_dataframe: Any = None,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
    no_cuda_streams: bool = False,
    track_mean_mode: Optional[str] = None,
    profiler: Any = None,
):
    mean_tracker: Optional[MeanTracker] = None
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
            get_timer = Timer()
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

            using_precalculated_tracking = (
                tracking_dataframe is not None and tracking_dataframe.has_input_data()
            )
            using_precalculated_detection = (
                detection_dataframe is not None and detection_dataframe.has_input_data()
            )
            using_precalculated_pose = pose_dataframe is not None and pose_dataframe.has_input_data()

            # Build AspenNet if a config is provided under config['aspen']
            aspen_net: Optional[AspenNet] = None
            aspen_cfg: Optional[Dict[str, Any]] = None
            cfg_aspen = config.get("aspen")
            if isinstance(cfg_aspen, dict):
                aspen_cfg = dict(cfg_aspen)
            if aspen_cfg:
                trunks_cfg = aspen_cfg.get("trunks", {}) or {}
                initial_args = config.get("initial_args", {}) or {}

                # Dynamically disable pose trunk if not requested, unless loading pose or
                # a downstream trunk (e.g., pose_to_det) requires it.
                # if "pose" in trunks_cfg and not using_precalculated_pose:
                #     requires_pose = any(name in trunks_cfg for name in ("pose_to_det", "pose_bbox_adapter"))
                #     if not requires_pose:
                #         trunks_cfg["pose"]["enabled"] = False

                # Replace compute trunks with loader variants if using precomputed data
                changes = []
                if "detector" in trunks_cfg and using_precalculated_detection:
                    trunks_cfg["detector"]["class"] = "hmlib.aspen.trunks.load.LoadDetectionsTrunk"
                    if "detector_factory" in trunks_cfg:
                        trunks_cfg["detector_factory"]["enabled"] = False
                    changes.append("swap detector->LoadDetectionsTrunk; disable detector_factory")
                if "tracker" in trunks_cfg and using_precalculated_tracking:
                    trunks_cfg["tracker"]["class"] = "hmlib.aspen.trunks.load.LoadTrackingTrunk"
                    if "model_factory" in trunks_cfg:
                        trunks_cfg["model_factory"]["enabled"] = False
                    if "boundaries" in trunks_cfg:
                        trunks_cfg["boundaries"]["enabled"] = False
                    changes.append("swap tracker->LoadTrackingTrunk; disable model_factory,boundaries")
                if "pose" in trunks_cfg and using_precalculated_pose:
                    trunks_cfg["pose"]["class"] = "hmlib.aspen.trunks.load.LoadPoseTrunk"
                    changes.append("swap pose->LoadPoseTrunk")

                # Default to saving when not loading; explicit flags still honored by presence of dataframes
                save_detection = not using_precalculated_detection
                save_tracking = not using_precalculated_tracking
                save_pose = not using_precalculated_pose

                def _append_trunk(name: str, cls: str, depends_on: str):
                    if name not in trunks_cfg:
                        trunks_cfg[name] = {"class": cls, "depends": [depends_on], "params": {}}

                if "detector" in trunks_cfg and save_detection and trunks_cfg.get("detector", {}).get("enabled", True):
                    _append_trunk("save_detections", "hmlib.aspen.trunks.save.SaveDetectionsTrunk", "detector")
                    changes.append("append save_detections")
                if "tracker" in trunks_cfg and save_tracking and trunks_cfg.get("tracker", {}).get("enabled", True):
                    _append_trunk("save_tracking", "hmlib.aspen.trunks.save.SaveTrackingTrunk", "tracker")
                    changes.append("append save_tracking")
                if "pose" in trunks_cfg and save_pose and trunks_cfg.get("pose", {}).get("enabled", True):
                    _append_trunk("save_pose", "hmlib.aspen.trunks.save.SavePoseTrunk", "pose")
                    changes.append("append save_pose")

                if changes:
                    logger.info("Aspen trunk patch: " + ", ".join(changes))

                aspen_cfg["trunks"] = trunks_cfg

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
                shared = dict(
                    model=model,
                    pose_inferencer=pose_inferencer,
                    postprocessor=postprocessor,
                    fp16=fp16,
                    device=device,
                    using_precalculated_tracking=using_precalculated_tracking,
                    using_precalculated_detection=using_precalculated_detection,
                    tracking_dataframe=tracking_dataframe,
                    detection_dataframe=detection_dataframe,
                    pose_dataframe=pose_dataframe,
                    action_dataframe=action_dataframe,
                    plot_pose=bool(config.get("plot_pose", False)),
                    # Propagate CLI flag to BoundariesTrunk -> IceRinkSegmBoundaries(draw)
                    plot_ice_mask=bool(config.get("plot_ice_mask", False)),
                    # Boundary + identity context for trunks
                    game_id=config.get("game_id"),
                    original_clip_box=config.get("original_clip_box"),
                    top_border_lines=config.get("top_border_lines"),
                    bottom_border_lines=config.get("bottom_border_lines"),
                )
                if profiler is not None:
                    shared["profiler"] = profiler
                aspen_net = AspenNet(aspen_cfg, shared=shared)
            # Optional torch profiler context spanning the run
            prof_ctx = profiler if getattr(profiler, "enabled", False) else contextlib.nullcontext()
            with prof_ctx:
                for cur_iter, dataset_results in enumerate(dataloader_iterator):
                    origin_imgs, data, _, info_imgs, ids = dataset_results.pop("pano")
                    if fps:
                        data["fps"] = fps
                    with torch.no_grad():
                        frame_id = info_imgs[2][0]

                    batch_size = origin_imgs.shape[0]

                    if last_frame_id is None:
                        last_frame_id = int(frame_id)
                    else:
                        assert int(frame_id) == last_frame_id + batch_size
                        last_frame_id = int(frame_id)

                    batch_size = origin_imgs.shape[0]

                    if detect_timer is None:
                        detect_timer = Timer()

                    if aspen_net is not None:
                        # Execute the configured DAG
                        # Prepare per-iteration context
                        iter_context: Dict[str, Any] = dict(
                            origin_imgs=origin_imgs,
                            data=data,
                            ids=ids,
                            info_imgs=info_imgs,
                            frame_id=int(frame_id),
                            device=device,
                            cuda_stream=cuda_stream,
                            detect_timer=detect_timer,
                            mean_tracker=mean_tracker,
                            using_precalculated_tracking=using_precalculated_tracking,
                            using_precalculated_detection=using_precalculated_detection,
                        )
                        # Merge shared into context for trunks convenience
                        iter_context.update(aspen_net.shared)
                        if dataset_results:
                            iter_context.setdefault("data", {})["dataset_results"] = dataset_results

                        if getattr(profiler, "enabled", False):
                            with profiler.rf("aspen.forward"):
                                out_context = aspen_net(iter_context)
                        else:
                            out_context = aspen_net(iter_context)
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

                    # Removed legacy per-frame post-processing branch

                    if detect_timer is not None and cur_iter % 50 == 0:
                        # print(
                        logger.info(
                            "mmtrack tracking, frame {} ({:.2f} fps)".format(
                                frame_id,
                                batch_size * 1.0 / max(1e-5, detect_timer.average_time),
                            )
                        )
                        detect_timer = Timer()

                    if cur_iter % 200 == 0:
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
    except Exception as ex:
        raise
    finally:
        if tracking_dataframe is not None:
            tracking_dataframe.close()
        if detection_dataframe is not None:
            detection_dataframe.close()
        if pose_dataframe is not None:
            pose_dataframe.close()
        if mean_tracker is not None:
            mean_tracker.close()
