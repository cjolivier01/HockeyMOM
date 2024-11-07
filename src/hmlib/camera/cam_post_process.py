from __future__ import absolute_import, division, print_function

import argparse
import os
import time
import traceback
from threading import Thread
from typing import Any, Dict, Optional, Union

import torch

from hmlib.bbox.box_functions import aspect_ratio
from hmlib.builder import HM
from hmlib.camera.camera_data import CameraTrackingData
from hmlib.camera.play_tracker import PlayTracker
from hmlib.config import get_nested_value
from hmlib.log import logger
from hmlib.tracking_utils.timer import Timer, TimeTracker
from hmlib.ui import Shower
from hmlib.utils.containers import create_queue
from hmlib.utils.progress_bar import ProgressBar
from hmlib.video.video_out import VideoOutput
from hockeymom import core

# MAX_CROPPED_WIDTH = 4096
MAX_CROPPED_WIDTH = 7680


##
#  _____         __             _ _                                                     _
# |  __ \       / _|           | | |      /\                                           | |
# | |  | | ___ | |_  __ _ _   _| | |_    /  \    _ __  __ _ _   _ _ __ ___   ___  _ __ | |_  ___
# | |  | |/ _ \|  _|/ _` | | | | | __|  / /\ \  | '__|/ _` | | | | '_ ` _ \ / _ \| '_ \| __|/ __|
# | |__| |  __/| | | (_| | |_| | | |_  / ____ \ | |  | (_| | |_| | | | | | |  __/| | | | |_ \__ \
# |_____/ \___||_|  \__,_|\__,_|_|\__|/_/    \_\|_|   \__, |\__,_|_| |_| |_|\___||_| |_|\__||___/
#                                                      __/ |
#                                                     |___/
#
# Some experimental and debugging parameters that aid in development
#
class DefaultArguments(core.HMPostprocessConfig):
    def __init__(
        self,
        game_config: Dict,
        basic_debugging: int = 0,
        output_video_path: str = None,
        opts: argparse.Namespace = None,
    ):
        # basic_debugging = False
        self.debug = int(basic_debugging)

        super().__init__()

        self.game_config = game_config

        self._output_video_path = output_video_path

        # Display the image every frame (slow)
        self.show_image = self.show_image or basic_debugging

        # Draw individual player boxes, tracking ids, speed and history trails
        self.plot_individual_player_tracking = False or basic_debugging
        # self.plot_individual_player_tracking = True

        # Draw intermediate boxes which are used to compute the final camera box
        self.plot_cluster_tracking = False or basic_debugging
        # self.plot_cluster_tracking = True

        # Use a differenmt algorithm when fitting to the proper aspect ratio,
        # such that the box calculated is much larger and often takes
        # the entire height.  The drawback is there's not much zooming.
        self.max_in_aspec_ratio = True
        # self.max_in_aspec_ratio = False

        # Zooming is fixed based upon the horizonal position's distance from center
        # self.apply_fixed_edge_scaling = False
        self.apply_fixed_edge_scaling = True

        self.fixed_edge_scaling_factor = self.game_config["rink"]["camera"][
            "fixed_edge_scaling_factor"
        ]

        self.plot_moving_boxes = False or basic_debugging
        # self.plot_moving_boxes = True

        # self.old_tracking_use_new_moving_box = True
        self.old_tracking_use_new_moving_box = False

        # Print each frame number in the upper left corner
        self.plot_frame_number = False or basic_debugging
        # self.plot_frame_number = True

        self.plot_boundaries = False or basic_debugging or self.plot_individual_player_tracking
        # self.plot_boundaries = True

        # Plot frame ID and speed/velocity in upper-left corner
        self.plot_speed = False

        # self.fixed_edge_rotation = False
        self.fixed_edge_rotation = True

        self.fixed_edge_rotation_angle = self.game_config["rink"]["camera"][
            "fixed_edge_rotation_angle"
        ]

        # Plot the component shapes directly related to camera stickiness
        self.plot_sticky_camera = False or basic_debugging
        # self.plot_sticky_camera = True

        # self.cam_ignore_largest = self.game_config["rink"]["tracking"]["cam_ignore_largest"]
        self.cam_ignore_largest = get_nested_value(
            self.game_config, "rink.tracking.cam_ignore_largest", default_value=False
        )

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True and not basic_debugging
        # self.crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False

        # Draw watermark on the image
        self.use_watermark = True
        # self.use_watermark = False

        # Deprecated
        self.detection_inclusion_box = None

        self.skip_final_video_save = False

        #
        # Detection boundaries
        # TODO: Somehow move into boundaries class like witht he clip stuff
        # TODO: Get rid of this, probably no longer needed
        #
        self.top_border_lines = get_nested_value(self.game_config, "game.boundaries.upper", [])
        self.bottom_border_lines = get_nested_value(self.game_config, "game.boundaries.lower", [])
        upper_tune_position = get_nested_value(
            self.game_config, "game.boundaries.upper_tune_position", []
        )
        lower_tune_position = get_nested_value(
            self.game_config, "game.boundaries.lower_tune_position", []
        )
        boundary_scale_width = get_nested_value(
            self.game_config, "game.boundaries.scale_width", 1.0
        )
        boundary_scale_height = get_nested_value(
            self.game_config, "game.boundaries.scale_height", 1.0
        )
        if self.top_border_lines and upper_tune_position:
            for i in range(len(self.top_border_lines)):
                if boundary_scale_width:
                    self.top_border_lines[i][0] *= boundary_scale_width
                    self.top_border_lines[i][1] *= boundary_scale_width
                if boundary_scale_height:
                    self.top_border_lines[i][2] *= boundary_scale_height
                    self.top_border_lines[i][3] *= boundary_scale_height
                self.top_border_lines[i][0] += upper_tune_position[0]
                self.top_border_lines[i][2] += upper_tune_position[0]
                self.top_border_lines[i][1] += upper_tune_position[1]
                self.top_border_lines[i][3] += upper_tune_position[1]

        if self.bottom_border_lines and lower_tune_position:
            for i in range(len(self.top_border_lines)):
                if boundary_scale_width:
                    self.bottom_border_lines[i][0] *= boundary_scale_width
                    self.bottom_border_lines[i][1] *= boundary_scale_width
                if boundary_scale_height:
                    self.bottom_border_lines[i][3] *= boundary_scale_height
                    self.bottom_border_lines[i][2] *= boundary_scale_height
                self.bottom_border_lines[i][0] += lower_tune_position[0]
                self.bottom_border_lines[i][2] += lower_tune_position[0]
                self.bottom_border_lines[i][1] += lower_tune_position[1]
                self.bottom_border_lines[i][3] += lower_tune_position[1]

        if opts is not None:
            self.copy_args_if_not_exist(opts, self)

    @staticmethod
    def copy_args_if_not_exist(source, target):
        """
        Copy all attributes from source to target if they don't already exist in target.

        Parameters:
        - source: An object (e.g., argparse.Namespace) from which to copy attributes.
        - target: The target object to which attributes should be copied.
        """
        for attribute in vars(source):
            if not attribute.startswith("_"):
                if not hasattr(target, attribute):
                    setattr(target, attribute, getattr(source, attribute))


@HM.register_module()
class CamTrackPostProcessor:

    def __init__(
        self,
        hockey_mom,
        start_frame_id,
        data_type,
        fps: float,
        save_dir,
        output_video_path: Optional[str],
        device,
        original_clip_box,
        video_out_pipeline: Dict[str, Any],
        args: argparse.Namespace,
        save_frame_dir: str = None,
        async_post_processing: bool = False,
        video_out_device: str = None,
        video_out_cache_size: int = 2,
        async_video_out: bool = False,
        no_frame_postprocessing: bool = False,
        progress_bar: ProgressBar | None = None,
    ):
        self._args = args
        self._no_frame_postprocessing = no_frame_postprocessing
        self._start_frame_id = start_frame_id
        self._hockey_mom = hockey_mom
        self._async_video_out = async_video_out
        self._video_out_cache_size = video_out_cache_size
        self._video_out_pipeline = video_out_pipeline
        self._queue = create_queue(mp=False)
        self._data_type = data_type
        self._fps = fps
        self._thread = None
        self._final_aspect_ratio = torch.tensor(16.0 / 9.0, dtype=torch.float)
        self._output_video = None
        self._async_post_processing = async_post_processing
        self._timer = Timer()
        self._video_out_device = video_out_device
        self._original_clip_box = original_clip_box

        if self._video_out_device is None:
            self._video_out_device = device

        self._save_dir = save_dir
        self._save_frame_dir = save_frame_dir
        self._output_video_path = output_video_path

        self._camera_tracking_data = None

        self._video_output_campp = None
        self._queue_timer = Timer()
        self._send_to_timer_post_process = Timer()
        self._exception = None
        self._shower: Union[None, Shower] = None
        self._play_tracker = PlayTracker(
            hockey_mom=hockey_mom,
            play_box=hockey_mom._video_frame.bounding_box(),
            device=device,
            original_clip_box=original_clip_box,
            progress_bar=progress_bar,
            args=args,
        )
        self.secondary_init()

    def secondary_init(self):
        if self._args.crop_output_image:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = self._hockey_mom.video.height * self._final_aspect_ratio
            if self.final_frame_width > MAX_CROPPED_WIDTH:
                self.final_frame_width = MAX_CROPPED_WIDTH
                self.final_frame_height = self.final_frame_width / self._final_aspect_ratio

        else:
            self.final_frame_height = self._hockey_mom.video.height
            self.final_frame_width = self._hockey_mom.video.width

        self.final_frame_width = int(self.final_frame_width + 0.5)
        self.final_frame_height = int(self.final_frame_height + 0.5)

        if self._args.save_camera_data and self._save_dir:
            self._camera_tracking_data = CameraTrackingData(
                output_file=os.path.join(self._save_dir, "camera.csv")
            )

        if not self._no_frame_postprocessing and self.output_video_path:
            assert self._video_output_campp is None
            self._video_output_campp = VideoOutput(
                name="TRACKING",
                args=self._args,
                output_video_path=self.output_video_path,
                fps=self._fps,
                start=False,
                output_frame_width=self.final_frame_width,
                output_frame_height=self.final_frame_height,
                save_frame_dir=self._save_frame_dir,
                original_clip_box=self._original_clip_box,
                cache_size=self._video_out_cache_size,
                async_output=self._async_video_out,
                video_out_pipeline=self._video_out_pipeline,
                watermark_image_path=(
                    os.path.realpath(
                        os.path.join(
                            os.path.dirname(os.path.realpath(__file__)),
                            "..",
                            "..",
                            "..",
                            "images",
                            "sports_ai_watermark.png",
                        )
                    )
                    if self._args.use_watermark
                    else None
                ),
                device=self._video_out_device,
                skip_final_save=self._args.skip_final_video_save,
                image_channel_adjustment=self._args.game_config["rink"]["camera"][
                    "image_channel_adjustment"
                ],
            )
            self._video_output_campp.start()
        elif self._args.show_image:
            self._shower = Shower("CamTrackPostProcessor", self._args.show_scaled, max_size=1)

    def eval(self):
        self._play_tracker.eval()

    @property
    def output_video_path(self):
        return self._output_video_path

    def start(self):
        self._thread = Thread(target=self._start, name="CamPostProc")
        self._thread.start()

    def _start(self):
        return self.postprocess_frame_worker()

    def stop(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
        self._video_output_campp.stop()

    def send(
        self,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self._exception is not None:
            raise self._exception
        try:
            if self._async_post_processing:
                with TimeTracker(
                    "Send to cam post process queue",
                    self._send_to_timer_post_process,
                    print_interval=50,
                ):
                    wait_count = 0
                    while self._queue.qsize() > 1:
                        if not self._args.debug and not self._args.show_image:
                            wait_count += 1
                            if wait_count % 100 == 0:
                                logger.info("Cam post-process queue too large")
                        time.sleep(0.001)
                    self._queue.put(data)
            else:
                with torch.no_grad():
                    results = self._play_tracker.forward(results=data)
                del data
                for frame_id, current_box in zip(results["frame_ids"], results["current_box"]):
                    assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                    if self._camera_tracking_data is not None:
                        self._camera_tracking_data.add_frame_records(
                            frame_id=frame_id,
                            tlbr=current_box if current_box.ndim == 4 else current_box.unsqueeze(0),
                        )
                if self._video_output_campp is not None:
                    self._video_output_campp.append(results)
                elif self._shower is not None and "img" in results:
                    self._shower.show(results["img"].cpu())
                return results
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def postprocess_frame_worker(self):
        try:
            self._postprocess_frame_worker()
        except Exception as ex:
            self._exception = ex
            print(ex)
            traceback.print_exc()
            raise
        finally:
            if self._video_output_campp is not None:
                self._video_output_campp.stop()

    def _postprocess_frame_worker(self):
        while True:
            results = self._queue.get()
            if results is None:
                break

            with torch.no_grad():
                results = self._play_tracker.forward(results=results)

            for frame_id, current_box in zip(results["frame_ids"], results["current_box"]):
                assert torch.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                if self._camera_tracking_data is not None:
                    self._camera_tracking_data.add_frame_records(
                        frame_id=frame_id,
                        tlbr=current_box if current_box.ndim == 4 else current_box.unsqueeze(0),
                    )
            if self._video_output_campp is not None:
                self._video_output_campp.append(results)

    def get_arena_box(self):
        return self._hockey_mom._video_frame.bounding_box()
