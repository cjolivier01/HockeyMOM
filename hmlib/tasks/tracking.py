from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from mmengine.structures import InstanceData

# from mmcv.parallel import collate, scatter
from torch.cuda.amp import autocast

import hmlib.models.end_to_end  # Registers the model
import hmlib.tracking_utils.segm_boundaries
from hmlib.log import logger
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
from hmlib.tracking_utils.timer import Timer
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils import MeanTracker
from hmlib.utils.gpu import StreamTensor, copy_gpu_to_gpu_async, cuda_stream_scope
from hmlib.utils.image import make_channels_first, make_channels_last
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, convert_seconds_to_hms

# AspenNet graph runner
from hmlib.aspen import AspenNet

from .multi_pose import multi_pose_task


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
                # Dynamically disable pose trunk if not requested
                if not bool(config.get("multi_pose", False)) and "trunks" in aspen_cfg and "pose" in aspen_cfg["trunks"]:
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
                        # Legacy linear pipeline
                        # TODO: maybe make f16/bf16?
                        # So far, tracking goes to Hell for some reason when using 16-bit,
                        # maybe the Kalman filter. Seems like detection itself should be good
                        # enough at 16-bit, so maybe need to modify that
                        # ByteTrack does a "batch" of just one, but the second dim
                        # is the # frames, so it's like a batch, but the frames of the same
                        # video are the batch items.  This is why we unsqueeze here.
                        detection_image = data["img"]
                        detection_image = make_channels_first(detection_image)
                        if isinstance(detection_image, StreamTensor):
                            detection_image.verbose = True
                            detection_image = detection_image.wait(cuda_stream)

                        if detection_image.device != device:
                            detection_image, _ = copy_gpu_to_gpu_async(tensor=detection_image, dest_device=device)

                        # Make batch size of 1, but some T number of frames (prev batch size)
                        detection_image = detection_image.unsqueeze(0)
                        assert detection_image.ndim == 5

                        if "original_images" not in data:
                            data["original_images"] = origin_imgs

                        if dataset_results:
                            data["dataset_results"] = dataset_results

                        if mean_tracker is not None:
                            detection_image = mean_tracker.forward(detection_image)

                        # forward the model
                        detect_timer.tic()
                        with torch.no_grad():
                            with autocast() if fp16 else nullcontext():
                                data["img"] = detection_image
                                detection_image = None
                                data = model(return_loss=False, rescale=True, **data)

                        detect_timer.toc()
                        track_data_sample = data["data_samples"]
                        nr_tracks = int(track_data_sample.video_data_samples[0].metainfo["nr_tracks"])
                        tracking_ids = track_data_sample.video_data_samples[-1].pred_track_instances.instances_id
                        if len(tracking_ids):
                            max_tracking_id = torch.max(tracking_ids)

                        if True:
                            jersey_results = data.get("jersey_results")
                            for frame_index, video_data_sample in enumerate(
                                track_data_sample.video_data_samples
                            ):
                                pred_track_instances = getattr(
                                    video_data_sample, "pred_track_instances", None
                                )
                                if pred_track_instances is None:
                                    # we arent tracking anything (probably a performance test)
                                    cuda_stream.synchronize()
                                    continue

                                if not using_precalculated_tracking:
                                    if tracking_dataframe is not None:
                                        tracking_dataframe.add_frame_records(
                                            frame_id=frame_id + frame_index,
                                            tracking_ids=pred_track_instances.instances_id,
                                            tlbr=pred_track_instances.bboxes,
                                            scores=pred_track_instances.scores,
                                            labels=pred_track_instances.labels,
                                            jersey_info=(
                                                jersey_results[frame_index]
                                                if jersey_results is not None
                                                else None
                                            ),
                                        )
                                if not using_precalculated_detection:
                                    if detection_dataframe is not None:
                                        detection_dataframe.add_frame_records(
                                            frame_id=frame_id,
                                            scores=video_data_sample.pred_instances.scores,
                                            labels=video_data_sample.pred_instances.labels,
                                            bboxes=video_data_sample.pred_instances.bboxes,
                                        )

                            # Clean data to send of the batched images
                            data_to_send = data.copy()
                            del data["original_images"]
                            if "img" in data_to_send:
                                del data_to_send["img"]

                            if not using_precalculated_tracking:
                                if pose_inferencer is not None:
                                    if isinstance(data_to_send["original_images"], StreamTensor):
                                        data_to_send["original_images"] = data_to_send["original_images"].wait()
                                    pose_results = multi_pose_task(
                                        pose_inferencer=pose_inferencer,
                                        cur_frame=data_to_send["original_images"],
                                        show=config["plot_pose"],
                                    )
                                    data_to_send["pose_results"] = pose_results

                            if postprocessor is not None:
                                results = postprocessor.process_tracking(results=data_to_send)
                                results = None

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


def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if "track_bboxes" in mmtracking_results:
        tracking_results = mmtracking_results["track_bboxes"][0]
    elif "track_results" in mmtracking_results:
        tracking_results = mmtracking_results["track_results"][0]

    for track in tracking_results:
        person = {}
        person["track_id"] = int(track[0])
        person["bbox"] = track[1:]  # will also have the score
        person_results.append(person)
    return person_results
