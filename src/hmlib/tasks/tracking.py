from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmcv.ops import RoIPool
from mmdet.structures import DetDataSample, TrackDataSample

# from mmcv.parallel import collate, scatter
from torch.cuda.amp import autocast

import hmlib.models.end_to_end  # Registers the model
import hmlib.tracking_utils.segm_boundaries
from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.utils.gpu import StreamTensor, copy_gpu_to_gpu_async
from hmlib.utils.image import make_channels_first, make_channels_last
from hmlib.utils.iterators import CachedIterator
from hmlib.utils.progress_bar import ProgressBar, convert_seconds_to_hms

from .multi_pose import multi_pose_task


def run_mmtrack(
    model,
    pose_model,
    pose_dataset_type,
    pose_dataset_info,
    config: Dict[str, Any],
    dataloader,
    postprocessor,
    progress_bar: Optional[ProgressBar] = None,
    tracking_dataframe: TrackingDataFrame = None,
    device: torch.device = None,
    input_cache_size: int = 2,
    fp16: bool = False,
):
    try:
        cuda_stream = torch.cuda.Stream(device)
        with torch.cuda.stream(cuda_stream):
            dataloader_iterator = CachedIterator(
                iterator=iter(dataloader), cache_size=input_cache_size
            )
            # print("WARNING: Not cacheing data loader")

            #
            # Calculate some dataset stats for our progress display
            #
            batch_size = dataloader.batch_size
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
            for cur_iter, dataset_results in enumerate(dataloader_iterator):
                origin_imgs, data, _, info_imgs, ids = dataset_results.pop("pano")
                with torch.no_grad():
                    frame_id = info_imgs[2][0]

                    batch_size = origin_imgs.shape[0]

                    if last_frame_id is None:
                        last_frame_id = int(frame_id)
                    else:
                        assert int(frame_id) == last_frame_id + batch_size
                        last_frame_id = int(frame_id)

                    batch_size = origin_imgs.shape[0]

                    if not using_precalculated_tracking:
                        if detect_timer is None:
                            detect_timer = Timer()

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
                            # detection_image = detection_image.get()
                            detection_image = detection_image.wait()

                        if detection_image.device != device:
                            detection_image, _ = copy_gpu_to_gpu_async(
                                tensor=detection_image, dest_device=device
                            )
                            # detection_image = detection_image.to(device, non_blocking=True)

                        # Make batch size of 1, but some T number of frames (prev batch size)
                        detection_image = detection_image.unsqueeze(0)
                        assert detection_image.ndim == 5

                        if "original_images" not in data:
                            data["original_images"] = origin_imgs

                        if dataset_results:
                            data["dataset_results"] = dataset_results

                        # forward the model
                        detect_timer.tic()
                        with torch.no_grad():
                            with autocast() if fp16 else nullcontext():
                                data["img"] = detection_image
                                detection_image = None
                                data = model(return_loss=False, rescale=True, **data)

                        detect_timer.toc()
                        # del data["img"]
                        # del detection_image
                        track_data_sample = data["data_samples"]
                        # assert len(tracking_results) == 1
                        # track_data_sample: TrackDataSample = tracking_results[0]
                        nr_tracks = int(
                            track_data_sample.video_data_samples[0].metainfo["nr_tracks"]
                        )
                        tracking_ids = track_data_sample.video_data_samples[
                            -1
                        ].pred_track_instances.instances_id
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
                                        frame_id=frame_id,
                                        tracking_ids=pred_track_instances.instances_id,
                                        tlbr=pred_track_instances.bboxes,
                                        scores=pred_track_instances.scores,
                                        jersey_info=(
                                            jersey_results[frame_index]
                                            if jersey_results is not None
                                            else None
                                        ),
                                    )
                            else:
                                # track_ids, scores, bboxes = (
                                #     tracking_dataframe.get_tracking_info_by_frame(frame_id)
                                # )
                                # online_tlwhs = torch.from_numpy(bboxes)
                                # tracking_results = None
                                assert False

                        # Clean data to send of the batched images
                        data_to_send = data.copy()
                        del data["original_images"]
                        if "img" in data_to_send:
                            del data_to_send["img"]

                        if postprocessor is not None:
                            # if isinstance(origin_imgs, StreamTensor):
                            #     origin_imgs.verbose = True
                            #     origin_imgs = origin_imgs.get()
                            # tracking_results, detections, online_tlwhs = (
                            results = postprocessor.process_tracking(results=data_to_send)
                            results = None
                    else:
                        for frame_index, frame_id in enumerate(info_imgs[2]):

                            video_data_samples = track_data_sample.video_data_samples[frame_index]
                            pred_track_instances = getattr(
                                video_data_samples, "pred_track_instances", None
                            )
                            if pred_track_instances is None:
                                # we arent tracking anything (probably a performance test)
                                cuda_stream.synchronize()
                                continue

                            if not using_precalculated_tracking:
                                if pose_model is not None:
                                    (
                                        tracking_results,
                                        pose_results,
                                        returned_outputs,
                                        vis_frame,
                                    ) = multi_pose_task(
                                        config=config,
                                        pose_model=pose_model,
                                        cur_frame=make_channels_last(origin_imgs).wait().squeeze(0),
                                        dataset=pose_dataset_type,
                                        dataset_info=pose_dataset_info,
                                        tracking_results=tracking_results,
                                        smooth=config["smooth"],
                                        show=config["plot_pose"],
                                    )
                                    if isinstance(vis_frame, np.ndarray):
                                        vis_frame = torch.from_numpy(vis_frame)
                                    if isinstance(origin_imgs, StreamTensor):
                                        origin_imgs = origin_imgs.wait()
                                    origin_imgs[frame_index] = vis_frame.to(
                                        device=origin_imgs.device, non_blocking=True
                                    )
                                else:
                                    vis_frame = None

                                track_ids = pred_track_instances.instances_id
                                bboxes = pred_track_instances.bboxes
                                scores = pred_track_instances.scores

                                if tracking_dataframe is not None:
                                    tracking_dataframe.add_frame_records(
                                        frame_id=frame_id,
                                        tracking_ids=track_ids,
                                        tlbr=bboxes,
                                        scores=scores,
                                    )

                                online_tlwhs = bboxes.clone()
                                # make boxes tlwh
                                online_tlwhs[:, 2] = (
                                    online_tlwhs[:, 2] - online_tlwhs[:, 0]
                                )  # width = x2 - x1
                                online_tlwhs[:, 3] = (
                                    online_tlwhs[:, 3] - online_tlwhs[:, 1]
                                )  # height = y2 - y1
                            else:
                                track_ids, scores, bboxes = (
                                    tracking_dataframe.get_tracking_info_by_frame(frame_id)
                                )
                                online_tlwhs = torch.from_numpy(bboxes)
                                tracking_results = None

                            online_ids = track_ids.clone()
                            online_scores = scores.clone()

                            # Clean data to send of the batched images
                            data_to_send = data.copy()
                            del data_to_send["original_images"]
                            if "img" in data_to_send:
                                del data_to_send["img"]

                            # sanity check that no batched tensors left
                            for _, val in data_to_send.items():
                                if isinstance(val, torch.Tensor):
                                    # make sure no batched tensors left
                                    assert val.ndim <= 3

                            if postprocessor is not None:
                                if isinstance(origin_imgs, StreamTensor):
                                    origin_imgs = origin_imgs.get()
                                tracking_results, detections, online_tlwhs = (
                                    postprocessor.process_tracking(
                                        tracking_results=tracking_results,
                                        frame_id=frame_id,
                                        online_tlwhs=online_tlwhs,
                                        online_ids=online_ids,
                                        online_scores=online_scores,
                                        detections=video_data_samples.pred_instances.bboxes,
                                        info_imgs=info_imgs,
                                        letterbox_img=None,
                                        original_img=origin_imgs[frame_index].unsqueeze(0),
                                        data=data_to_send,
                                    )
                                )

                            if pose_model is not None and using_precalculated_tracking:
                                if tracking_results is None:
                                    # Reconstruct tracking_results
                                    tracking_items = torch.from_numpy(
                                        track_ids.astype(bboxes.dtype)
                                    ).reshape(track_ids.shape[0], 1)
                                    tracking_items = torch.cat(
                                        [
                                            tracking_items,
                                            torch.from_numpy(bboxes),
                                            torch.from_numpy(scores).reshape(scores.shape[0], 1),
                                        ],
                                        dim=1,
                                    )
                                    tracking_results = {
                                        "track_bboxes": [tracking_items.numpy()],
                                    }

                                multi_pose_task(
                                    config=config,
                                    pose_model=pose_model,
                                    cur_frame=make_channels_last(origin_imgs[frame_index]),
                                    dataset=pose_dataset_type,
                                    dataset_info=pose_dataset_info,
                                    tracking_results=tracking_results,
                                    smooth=config["smooth"],
                                    show=config["show_image"],
                                    # show=True,
                                )
                        # end frame loop

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
