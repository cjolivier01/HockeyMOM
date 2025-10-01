from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import yaml

import hmlib.models.end_to_end  # Registers the model
import hmlib.tracking_utils.segm_boundaries

# AspenNet graph runner
from hmlib.aspen import AspenNet
from hmlib.log import logger
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
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
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
    no_cuda_streams: bool = False,
    track_mean_mode: Optional[str] = None,
):
    mean_tracker: Optional[MeanTracker] = None
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

            #
            # Build AspenNet if a config is provided
            #
            aspen_config_path: Optional[str] = config.get("aspen_config")
            aspen_net: Optional[AspenNet] = None
            if aspen_config_path:
                with open(aspen_config_path, "r") as f:
                    aspen_cfg = yaml.safe_load(f)
                # Dynamically disable pose trunk if not requested, unless
                # a downstream trunk (e.g., pose_to_det) requires it.
                if not bool(config.get("multi_pose", False)) and "trunks" in aspen_cfg and "pose" in aspen_cfg["trunks"]:
                    trunks_cfg = aspen_cfg.get("trunks", {}) or {}
                    requires_pose = any(name in trunks_cfg for name in ("pose_to_det", "pose_bbox_adapter"))
                    if not requires_pose:
                        aspen_cfg["trunks"]["pose"]["enabled"] = False
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
                    plot_pose=bool(config.get("plot_pose", False)),
                    # Boundary + identity context for trunks
                    game_id=config.get("game_id"),
                    original_clip_box=config.get("original_clip_box"),
                    top_border_lines=config.get("top_border_lines"),
                    bottom_border_lines=config.get("bottom_border_lines"),
                )
                aspen_net = AspenNet(aspen_cfg, shared=shared)
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
        if mean_tracker is not None:
            mean_tracker.close()
