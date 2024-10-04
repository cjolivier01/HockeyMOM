import time
from typing import Any, Dict, List, Optional, Tuple, Union

# For TopDownGetBboxCenterScale
from typeguard import typechecked


@typechecked
def multi_pose_task(
    pose_model,
    cur_frame,
    dataset,
    dataset_info,
    config: Dict[str, Any],
    tracking_results: dict,
    smooth: bool = False,
    show: bool = False,
):
    start = time.time()
    # build pose smoother for temporal refinement
    if smooth:
        smoother = Smoother(filter_cfg=config["smooth_filter_cfg"], keypoint_dim=2)
    else:
        smoother = None

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # keep the person class bounding boxes.
    person_results = process_mmtracking_results(tracking_results)

    from mmpose.apis import inference_top_down_pose_model, vis_pose_tracking_result
    from mmpose.core import Smoother

    # test a single image, with a list of bboxes.
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        cur_frame,
        person_results,
        bbox_thr=config["bbox_thr"],
        format="xyxy",
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names,
    )
    duration = time.time() - start

    if smoother:
        pose_results = smoother.smooth(pose_results)

    vis_frame = None
    # show the results
    if show:
        # assert cur_frame.size(0) == 1
        vis_frame = vis_pose_tracking_result(
            pose_model,
            cur_frame.squeeze(0).to("cpu").numpy(),
            pose_results,
            radius=config["radius"],
            thickness=config["thickness"],
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=config["kpt_thr"],
            show=False,
            # show=True,
        )
        # vis_frame = np.expand_dims(vis_frame, axis=0)
    # duration = time.time() - start
    # print(f"pose took {duration} seconds")
    return tracking_results, pose_results, returned_outputs, vis_frame
