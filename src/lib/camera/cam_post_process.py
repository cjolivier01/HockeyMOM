from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from numba import njit

import time
import os
import cv2
import argparse
import numpy as np
import traceback

import torch
import torchvision as tv

from threading import Thread
from multiprocessing import Queue

from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracker.multitracker import torch_device

from .camera import aspect_ratio, width, height, center, center_distance, center_x_distance, translate_box, make_box_at_center

from hockeymom import core

core.hello_world()

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

RINK_CONFIG = {
    "vallco": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "dublin": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "yerba_buena": {
        "fixed_edge_scaling_factor": 1.5,
    }
}

BASIC_DEBUGGING = True

class DefaultArguments(core.HMPostprocessConfig):
    def __init__(self, rink: str = "yerba_buena", args: argparse.Namespace = None):
        super().__init__()
        # Display the image every frame (slow)
        self.show_image = False or BASIC_DEBUGGING

        # Draw individual player boxes, tracking ids, speed and history trails
        self.plot_individual_player_tracking = False

        # Draw intermediate boxes which are used to compute the final camera box
        self.plot_cluster_tracking = False or BASIC_DEBUGGING

        self.plot_camera_tracking = False or BASIC_DEBUGGING

        # Plot frame ID and speed/velocity in upper-left corner
        self.plot_speed = False

        # Use a differenmt algorithm when fitting to the proper aspect ratio,
        # such that the box calculated is much larger and often takes
        # the entire height.  The drawback is there's not much zooming.
        self.max_in_aspec_ratio = True

        # Only apply zoom when the camera box is against
        # either the left or right edge of the video
        self.no_max_in_aspec_ratio_at_edges = False

        # Zooming is fixed based upon the horizonal position's distance from center
        self.apply_fixed_edge_scaling = True

        self.fixed_edge_scaling_factor = RINK_CONFIG[rink]["fixed_edge_scaling_factor"]

        self.fixed_edge_rotation = False

        #self.fixed_edge_rotation_angle = 35.0
        self.fixed_edge_rotation_angle = 45.0

        # Use "sticky" panning, where panning occurs in less frequent,
        # but possibly faster, pans rather than a constant
        # pan (which may appear tpo "wiggle")
        self.sticky_pan = True

        # Plot the component shapes directly related to camera stickiness
        self.plot_sticky_camera = False or BASIC_DEBUGGING

        # Skip some number of frames before post-processing. Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.skip_frame_count = 0

        # Moving right-to-left
        # self.skip_frame_count = 450

        # Stop at the given frame and (presumably) output the final video.
        # Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.stop_at_frame = 0
        # self.stop_at_frame = 30*30

        # Make the image the same relative dimensions as the initial image,
        # such that the highest possible resolution is available when the camera
        # box is either the same height or width as the original video image
        # (Slower, but better final quality)
        self.scale_to_original_image = True
        #self.scale_to_original_image = False

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True and not BASIC_DEBUGGING

        # Don't crop image, but performa of the calculations
        # except for the actual image manipulations
        self.fake_crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False

        # Draw watermark on the image
        self.use_watermark = True
        #self.use_watermark = False


def scale_box(box, from_img, to_img):
    from_sz = (from_img.shape[1], from_img.shape[0])
    to_sz = (to_img.shape[1], to_img.shape[0])
    w_scale = to_sz[1] / from_sz[1]
    h_scale = to_sz[0] / from_sz[0]
    new_box = [box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale]
    print(f"from={box} -> to={new_box}")
    return new_box


def make_scale_array(from_img, to_img):
    from_sz = (from_img.shape[1], from_img.shape[0])
    to_sz = (to_img.shape[1], to_img.shape[0])
    w_scale = to_sz[0] / from_sz[0]
    h_scale = to_sz[1] / from_sz[1]
    return np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)


class ImageProcData:
    def __init__(self, frame_id: int, img, current_box, save_dir: str):
        self.frame_id = frame_id
        self.img = img
        self.current_box = current_box.copy()
        self.save_dir = save_dir


class FramePostProcessor:
    def __init__(
        self,
        hockey_mom,
        start_frame_id,
        data_type,
        fps: float,
        save_dir,
        result_filename,
        show_image,
        opt,
        args: argparse.Namespace,
        use_fork: bool = False,
    ):
        self._args = args
        self._frame_id = start_frame_id - 1
        self._hockey_mom = hockey_mom
        self._queue = Queue()
        self._imgproc_queue = Queue()
        self._data_type = data_type
        self._save_dir = save_dir
        self._result_filename = result_filename
        self._show_image = show_image
        self._fps = fps
        self._opt = opt
        self._thread = None
        self._imgproc_thread = None
        self._use_fork = use_fork
        self._final_aspect_ratio = 16.0 / 9.0
        self._output_video = None
        self.watermark = cv2.imread(
            os.path.realpath(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "..",
                    "..",
                    "images",
                    "sports_ai_watermark.png",
                )
            ),
            cv2.IMREAD_UNCHANGED,
        )
        self.watermark_height = self.watermark.shape[0]
        self.watermark_width = self.watermark.shape[1]
        self.watermark_rgb_channels = self.watermark[:, :, :3]
        self.watermark_alpha_channel = self.watermark[:, :, 3]
        self.watermark_mask = cv2.merge(
            [
                self.watermark_alpha_channel,
                self.watermark_alpha_channel,
                self.watermark_alpha_channel,
            ]
        )

    def get_first_frame_id(self):
        return self._args.skip_frame_count

    def start(self):
        if self._use_fork:
            self._child_pid = os.fork()
            if not self._child_pid:
                self.device = torch.device(torch_device())
                self._start()
        else:
            self._thread = Thread(target=self._start)
            self._thread.start()
            self._imgproc_thread = Thread(target=self._start_final_image_processing)
            self._imgproc_thread.start()

    def _start(self):
        if self._args.fake_crop_output_image:
            self.crop_output_image = True
        return self.postprocess_frame(
            self._hockey_mom,
            self._save_dir,
            self._result_filename,
            self._show_image,
            self._opt,
        )

    def _start_final_image_processing(self):
        return self.final_image_processing()

    def stop(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None
            self._imgproc_queue.put(None)
            self._imgproc_thread.join()
            self._imgproc_thread = None

        elif self.use_fork:
            self._queue.put(None)
            if self._child_pid:
                os.waitpid(self._child_pid)

    def send(self, online_tlwhs, online_ids, image, original_img):
        while self._queue.qsize() > 10:
            time.sleep(0.001)
        self._queue.put((online_tlwhs.copy(), online_ids.copy(), image, original_img))

    def postprocess_frame(self, hockey_mom, save_dir, result_filename, show_image, opt):
        try:
            self._postprocess_frame_impl(
                hockey_mom, save_dir, result_filename, show_image, opt
            )
        except Exception as ex:
            print(ex)
            traceback.print_exc()
            raise

    def final_image_processing(self):
        ready_string = self._imgproc_queue.get()
        assert ready_string == "ready"
        plot_interias = False
        show_image_interval = 1
        skip_frames_before_show = 0
        timer = Timer()
        if self._output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self._output_video = cv2.VideoWriter(
                # filename=self._save_dir + "/../tracking_output.mov",
                filename=self._save_dir + "/../tracking_output.avi",
                fourcc=fourcc,
                fps=self._fps,
                frameSize=(self.final_frame_width, self.final_frame_height),
                isColor=True,
            )
            assert self._output_video.isOpened()
            self._output_video.set(cv2.CAP_PROP_BITRATE, 27000 * 1024)
        while True:
            imgproc_data = self._imgproc_queue.get()
            if imgproc_data is None:
                if self._output_video is not None:
                    self._output_video.release()
                break
            timer.tic()

            if imgproc_data.frame_id % 20 == 0:
                logger.info(
                    "Image Post-Processing frame {} ({:.2f} fps)".format(
                        imgproc_data.frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )

            current_box = imgproc_data.current_box
            online_im = imgproc_data.img

            if self._args.fixed_edge_rotation:
                rotation_point = [int(i) for i in center(current_box)]
                width_center = online_im.shape[1] / 2
                if rotation_point[0] < width_center:
                #     dist_from_center_pct = (width_center - rotation_point[0])/width_center
                     mult = -1
                else:
                #     dist_from_center_pct = (rotation_point[0] - width_center)/width_center
                     mult = 1
                #angle = float(self._args.fixed_edge_rotation_angle)* dist_from_center_pct * mult

                gaussian = 1 - self._hockey_mom.get_gaussian_y_from_image_x_position(rotation_point[0], wide=True)
                #print(f"gaussian={gaussian}")
                angle = self._args.fixed_edge_rotation_angle - self._args.fixed_edge_rotation_angle * gaussian
                angle *= mult
                #print(f"angle={angle}")
                rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1.0)
                online_im = cv2.warpAffine(online_im, rotation_matrix, (online_im.shape[1], online_im.shape[0]))

            if self._args.crop_output_image:
                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                # print(f"crop ar={aspect_ratio(current_box)}")
                intbox = [int(i) for i in current_box]
                x1 = intbox[0]
                y1 = intbox[1]
                y2 = intbox[3]
                x2 = int(x1 + int(float(y2 - y1) * self._final_aspect_ratio))
                # if y2 == online_im.shape[0]:
                #     # Possible to be off by one here sometimes
                #     print(f"y2 off by one")
                #     y2 == online_im.shape[0] - 1
                #     assert y1 > 0
                #     y1 -= 1
                # if x2 == online_im.shape[1]:
                #     print(f"x2 off by one")
                #     # Possible to be off by one here sometimes
                #     x2 == online_im.shape[1] - 1
                #     assert x1 > 0
                #     x1 -= 1
                # Sanity check clip dimensions
                # print(f"shape={online_im.shape}, x1={x1}, x2={x2}, y1={y1}, y2={y2}")
                assert y1 >= 0 and y2 >= 0 and x1 >= 0 and x2 >= 0
                assert y1 < online_im.shape[0] and y2 < online_im.shape[0]
                assert x1 < online_im.shape[1] and x2 < online_im.shape[1]
                # hh = y2 - y1
                # ww = x2 - x1
                # assert hh < self.final_frame_height
                # assert ww < self.final_frame_width

                if not self._args.fake_crop_output_image:
                    if self._args.use_cuda:
                        gpu_image = torch.Tensor(online_im)[
                            y1 : y2 + 1, x1 : x2 + 1, 0:3
                        ].to("cuda")
                        # gpu_image = torch.Tensor(online_im).to("cuda:1")
                        # gpu_image = gpu_image[y1:y2,x1:x2,0:3]
                        # gpu_image = cv2.cuda_GpuMat(online_im)
                        # gpu_image = cv2.cuda_GpuMat(gpu_image, (x1, y1, x2, y2))
                    else:
                        online_im = online_im[y1 : y2 + 1, x1 : x2 + 1, 0:3]
                if not self._args.fake_crop_output_image and (
                    online_im.shape[0] != self.final_frame_height
                    or online_im.shape[1] != self.final_frame_width
                ):
                    if self._args.use_cuda:
                        if tv_resizer is None:
                            tv_resizer = tv.transforms.Resize(
                                size=(
                                    int(self.final_frame_width),
                                    int(self.final_frame_height),
                                )
                            )
                        gpu_image = tv_resizer.forward(gpu_image)
                    else:
                        # image_ar = float(online_im.shape[1])/float(online_im.shape[0])
                        # if not np.isclose(image_ar, self._final_aspect_ratio):
                        #      print(f"Not close: {image_ar} vs {self._final_aspect_ratio}")
                        # #     assert False

                        online_im = cv2.resize(
                            online_im,
                            dsize=(
                                int(self.final_frame_width),
                                int(self.final_frame_height),
                            ),
                            interpolation=cv2.INTER_CUBIC,
                        )
                assert online_im.shape[0] == self.final_frame_height
                assert online_im.shape[1] == self.final_frame_width
                if self._args.use_cuda:
                    # online_im = gpu_image.download()
                    online_im = np.array(gpu_image.cpu().numpy(), np.uint8)
            #
            # Watermark
            #
            if self._args.use_watermark:
                y = int(online_im.shape[0] - self.watermark_height)
                x = int(
                    online_im.shape[1]
                    - self.watermark_width
                    - self.watermark_width / 10
                )
                online_im[
                    y : y + self.watermark_height, x : x + self.watermark_width
                ] = online_im[
                    y : y + self.watermark_height, x : x + self.watermark_width
                ] * (
                    1 - self.watermark_mask / 255.0
                ) + self.watermark_rgb_channels * (
                    self.watermark_mask / 255.0
                )
            # Output (and maybe show) the final image
            if (
                self._args.show_image
                and imgproc_data.frame_id >= skip_frames_before_show
            ):
                if imgproc_data.frame_id % show_image_interval == 0:
                    cv2.imshow("online_im", online_im)
                    cv2.waitKey(1)

            if plot_interias:
                vis.plot_kmeans_intertias(hockey_mom=hockey_mom)

            if imgproc_data.save_dir is not None:
                if self._output_video is not None:
                    self._output_video.write(online_im)
                else:
                    cv2.imwrite(
                        os.path.join(
                            imgproc_data.save_dir,
                            "{:05d}.png".format(imgproc_data.frame_id),
                        ),
                        online_im,
                    )
            timer.toc()

    def _postprocess_frame_impl(
        self, hockey_mom, save_dir, result_filename, show_image, opt
    ):
        last_temporal_box = None
        last_sticky_temporal_box = None
        last_dx_shrink_size = 0
        max_dx_shrink_size = 100
        center_dx_shift = 0
        # show_image_interval = 1
        # tv_resizer = None
        timer = Timer()

        remove_largest = True

        if self._args.crop_output_image and not self._args.fake_crop_output_image:
            self.final_frame_height = int(hockey_mom.video.height)
            self.final_frame_width = int(
                hockey_mom.video.height * self._final_aspect_ratio
            )
            if self._args.use_cuda:
                self.final_frame_height /= 2.25
                self.final_frame_width /= 2.25
        else:
            self.final_frame_height = int(hockey_mom.video.height)
            self.final_frame_width = int(hockey_mom.video.width)

        self._imgproc_queue.put("ready")
        while True:
            online_targets_and_img = self._queue.get()
            if online_targets_and_img is None:
                break
            timer.tic()
            self._frame_id += 1

            if self._frame_id % 20 == 0:
                logger.info(
                    "Post-Processing frame {} ({:.2f} fps)".format(
                        self._frame_id, 1.0 / max(1e-5, timer.average_time)
                    )
                )

            online_tlwhs = online_targets_and_img[0]
            online_ids = online_targets_and_img[1]
            img0 = online_targets_and_img[2]
            original_img = online_targets_and_img[3]

            if remove_largest:
                largest_index = -1
                largest_height = -1

                for i, t in enumerate(online_tlwhs):
                    h = t[3]
                    if h > largest_height:
                        largest_height = h
                        largest_index = i

                if largest_index >= 0:
                    del online_tlwhs[largest_index]
                    del online_ids[largest_index]

            hockey_mom.append_online_objects(online_ids, online_tlwhs)

            hockey_mom.reset_clusters()
            hockey_mom.calculate_clusters(n_clusters=2)
            hockey_mom.calculate_clusters(n_clusters=3)

            if show_image or save_dir is not None:
                if self._args.scale_to_original_image:
                    online_im = original_img
                else:
                    online_im = img0

                # fast_bounding_box = None

                # if self._frame_id == 0:
                #     fast_ids = []
                #     fast_online_tlwhs = []
                # else:
                #     fast_ids = hockey_mom.get_fast_ids()
                #     if fast_ids:
                #         fast_online_tlwhs = [hockey_mom.get_tlwh(id) for id in fast_ids]
                #         fast_bounding_box = hockey_mom.get_current_bounding_box(
                #             ids=fast_ids
                #         )

                these_online_speeds = []
                for id in online_ids:
                    these_online_speeds.append(hockey_mom.get_spatial_speed(id))

                fast_ids_set = None
                if self._args.plot_individual_player_tracking:
                    # Plot the player boxes
                    fast_ids_set = set(hockey_mom.get_fast_ids())
                    if fast_ids_set:
                        #print(fast_ids_set)
                        fast_ids = []
                        fast_tlwhs = []
                        fast_speeds = []
                        for i, id in enumerate(online_ids):
                            if id in fast_ids_set:
                                fast_ids.append(id)
                                fast_tlwhs.append(online_tlwhs[i])
                                fast_speeds.append(these_online_speeds[i])
                        # online_ids = fast_ids
                        # online_tlwhs = fast_tlwhs
                        # these_online_speeds = fast_speeds
                        # online_im = vis.plot_tracking(
                        #     online_im,
                        #     fast_tlwhs,
                        #     fast_ids,
                        #     frame_id=self._frame_id,
                        #     fps=1.0 / timer.average_time if timer.average_time else 1000.0,
                        #     speeds=fast_speeds,
                        # )

                    online_im = vis.plot_tracking(
                        online_im,
                        online_tlwhs,
                        online_ids,
                        frame_id=self._frame_id,
                        fps=1.0 / timer.average_time if timer.average_time else 1000.0,
                        speeds=these_online_speeds,
                        line_thickness=1 if not fast_ids_set else 4,
                    )

                # Examine as 2 clusters
                largest_cluster_ids_2 = hockey_mom.prune_not_in_largest_cluster(
                    n_clusters=2, ids=online_ids
                )
                if largest_cluster_ids_2:
                    largest_cluster_ids_box2 = hockey_mom.get_current_bounding_box(
                        largest_cluster_ids_2
                    )
                    if self._args.plot_cluster_tracking:
                        vis.plot_rectangle(
                            online_im,
                            largest_cluster_ids_box2,
                            color=(128, 0, 0),  # dark red
                            thickness=6,
                            label="largest_cluster_ids_box2",
                        )
                else:
                    largest_cluster_ids_box2 = None

                # if fast_ids_set:
                #     print("Skipping 2-cluster box")
                #     largest_cluster_ids_box2 = None

                # Examine as 3 clusters
                largest_cluster_ids_3 = hockey_mom.prune_not_in_largest_cluster(
                    n_clusters=3, ids=online_ids
                )
                if largest_cluster_ids_3:
                    largest_cluster_ids_box3 = hockey_mom.get_current_bounding_box(
                        largest_cluster_ids_3
                    )
                    if self._args.plot_cluster_tracking:
                        vis.plot_rectangle(
                            online_im,
                            largest_cluster_ids_box3,
                            color=(0, 0, 128),  # dark blue
                            thickness=6,
                            label="largest_cluster_ids_box3",
                        )
                else:
                    largest_cluster_ids_box3 = None

                # if fast_ids_set:
                #     print("Skipping 3-cluster box")
                #     largest_cluster_ids_box3 = None

                if (
                    largest_cluster_ids_box2 is not None
                    and largest_cluster_ids_box3 is not None
                ):
                    current_box = hockey_mom.union_box(
                        largest_cluster_ids_box2, largest_cluster_ids_box3
                    )
                elif largest_cluster_ids_box2 is not None:
                    current_box = largest_cluster_ids_box2
                elif largest_cluster_ids_box3 is not None:
                    current_box = largest_cluster_ids_box3
                else:
                    current_box = hockey_mom._video_frame.box()

                # current_box = hockey_mom.ratioed_expand(current_box)
                if current_box is None:
                    current_box = hockey_mom._video_frame.box()

                outside_expanded_box = current_box + np.array([-100., -100., 100., 100.], dtype=np.float32)

                # if self._args.plot_camera_tracking:
                #     vis.plot_rectangle(
                #         online_im,
                #         current_box,
                #         color=(128, 0, 128),
                #         thickness=2,
                #         label="union_clusters_2_and_3",
                #     )

                # if fast_bounding_box is not None and self._args.plot_camera_tracking:
                #     vis.plot_rectangle(
                #         online_im,
                #         fast_bounding_box,
                #         color=(0, 255, 0),
                #         thickness=6,
                #         label="fast_bounding_box",
                #     )
                #     current_box = hockey_mom.union_box(current_box, fast_bounding_box)

                # current_box = scale(current_box, 2.25, 2.25)
                # current_box = hockey_mom.clamp(current_box)

                # if self._args.plot_camera_tracking:
                #     vis.plot_rectangle(
                #         online_im,
                #         current_box,
                #         color=(64, 64, 64),  # Dark gray
                #         thickness=6,
                #         label="scaled_union_clusters_2_and_3",
                #     )

                if self._args.plot_speed:
                    vis.plot_frame_id_and_speeds(
                        online_im,-
                        self._frame_id,
                        *hockey_mom.get_velocity_and_acceleratrion_xy(),
                    )

                def _apply_temporal(current_box, last_box, scale_speed: float, grays_level: int = 128, verbose: bool = False):
                    #
                    # Temporal: Apply velocity and acceleration
                    #
                    #nonlocal current_box, self
                    nonlocal self
                    current_box = hockey_mom.get_next_temporal_box(
                        current_box, last_box, scale_speed=scale_speed, verbose=verbose,
                    )
                    last_box = current_box.copy()
                    if self._args.plot_camera_tracking:
                        vis.plot_rectangle(
                            online_im,
                            current_box,
                            color=(grays_level, grays_level, grays_level),
                            thickness=2,
                            label="next_temporal_box",
                        )
                        # cv2.circle(
                        #     online_im,
                        #     [int(i) for i in center(current_box)],
                        #     radius=25,
                        #     color=(0, 0, 0),
                        #     thickness=25,
                        # )
                    return current_box, last_box

                group_x_velocity, edge_center = hockey_mom.get_group_x_velocity()
                if group_x_velocity:
                    #print(f"group x velocity: {group_x_velocity}")
                    # cv2.circle(
                    #     online_im,
                    #     _to_int(edge_center),
                    #     radius=30,
                    #     color=(255, 0, 255),
                    #     thickness=20,
                    # )
                    current_box = make_box_at_center(edge_center, width(current_box), height(current_box))
                    hockey_mom._current_camera_box_speed_x += group_x_velocity/2
                    #last_temporal_box = translate_box(last_temporal_box, group_x_velocity, 0)
                    #hockey_mom.add_x_velocity(group_x_velocity)
                    #hockey_mom.apply_box_velocity(current_box, scale_speed = 1.0)
                    #current_box = translate_box(current_box, group_x_velocity * 100, 0)
                    # vis.plot_rectangle(
                    #     online_im,
                    #     current_box,
                    #     color=(0, 0, 0),
                    #     thickness=10,
                    #     label="clamped_pre_aspect",
                    # )

                #current_box = hockey_mom.smooth_resize_box(current_box, last_temporal_box)
                current_box, last_temporal_box = _apply_temporal(current_box, last_temporal_box, scale_speed=1.0)

                # vis.plot_rectangle(
                #     online_im,
                #     outside_expanded_box,
                #     color=(0, 0, 0),
                #     thickness=5,
                #     label="",
                # )
                if not group_x_velocity:
                    hockey_mom.curtail_velocity_if_outside_box(
                        current_box, outside_expanded_box
                    )

                #
                # Aspect Ratio
                #
                # current_box = hockey_mom.clamp(current_box)
                fine_tracking_box = current_box.copy()
                if self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(0, 0, 0),
                        thickness=10,
                        label="clamped_pre_aspect",
                    )
                    # vis.plot_rectangle(
                    #     online_im,
                    #     current_box,
                    #     color=(0, 0, 0),
                    #     thickness=10,
                    #     label="clamped_pre_aspect",
                    # )

                current_box = hockey_mom.make_box_proper_aspect_ratio(
                    frame_id=self._frame_id,
                    the_box=current_box,
                    desired_aspect_ratio=self._final_aspect_ratio,
                    max_in_aspec_ratio=self._args.max_in_aspec_ratio,
                    no_max_in_aspec_ratio_at_edges=self._args.no_max_in_aspec_ratio_at_edges,
                )
                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

                # current_box = hockey_mom.clamp(current_box)

                # last_temporal_box = current_box.copy()
                # current_box = _apply_temporal()

                # print(f"make_box_proper_aspect_ratio ar={aspect_ratio(current_box)}")
                current_box = hockey_mom.shift_box_to_edge(current_box)
                # print(f"shift_box_to_edge ar={aspect_ratio(current_box)}")
                if self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(255, 255, 255),  # White
                        thickness=1,
                        label="after-aspect",
                    )
                if (
                    self._args.max_in_aspec_ratio
                    and self._args.no_max_in_aspec_ratio_at_edges
                ):
                    ZOOM_SHRINK_SIZE_INCREMENT = 1
                    box_is_at_right_edge = hockey_mom.is_box_at_right_edge(current_box)
                    box_is_at_left_edge = hockey_mom.is_box_at_left_edge(current_box)
                    cb_center = center(current_box)
                    if box_is_at_right_edge:
                        lt_center = center(last_temporal_box)
                        # frame_center = center(hockey_mom._video_frame.box())
                        if cb_center[0] < lt_center[0]:
                            last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT
                        elif cb_center[0] > lt_center[0]:
                            last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT
                    elif box_is_at_left_edge:
                        lt_center = center(last_temporal_box)
                        # frame_center = center(hockey_mom._video_frame.box())
                        if cb_center[0] > lt_center[0]:
                            last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT
                        elif cb_center[0] < lt_center[0]:
                            last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT
                    else:
                        # When not on edge, decay away the shrink sizes
                        # TODO: do with min/max
                        if last_dx_shrink_size > 0:
                            last_dx_shrink_size -= ZOOM_SHRINK_SIZE_INCREMENT * 1.5
                            if last_dx_shrink_size < 0:
                                last_dx_shrink_size = 0
                        elif last_dx_shrink_size < 0:
                            last_dx_shrink_size += ZOOM_SHRINK_SIZE_INCREMENT * 1.5
                            if last_dx_shrink_size > 0:
                                last_dx_shrink_size = 0

                    if last_dx_shrink_size > max_dx_shrink_size:
                        last_dx_shrink_size = max_dx_shrink_size
                    elif last_dx_shrink_size < -max_dx_shrink_size:
                        last_dx_shrink_size = -max_dx_shrink_size
                    if True:  # last_dx_shrink_size:
                        # print(f"Shrink width: {last_dx_shrink_size}")
                        w = width(current_box)
                        w -= last_dx_shrink_size
                        if box_is_at_right_edge:
                            center_dx_shift += 2
                            if center_dx_shift > last_dx_shrink_size:
                                center_dx_shift = last_dx_shrink_size
                            cb_center[0] += center_dx_shift
                        elif box_is_at_left_edge:
                            center_dx_shift -= 2
                            if center_dx_shift < -last_dx_shrink_size:
                                center_dx_shift = -last_dx_shrink_size
                        else:
                            if center_dx_shift < 0:
                                center_dx_shift += 2
                                if center_dx_shift > 0:
                                    center_dx_shift = 0
                            elif center_dx_shift > 0:
                                center_dx_shift -= 2
                                if center_dx_shift < 0:
                                    center_dx_shift = 0
                        cb_center[0] += center_dx_shift
                        h = w / self._final_aspect_ratio
                        current_box = np.array(
                            (
                                cb_center[0] - (w / 2.0) + 0.5,
                                cb_center[1] - (h / 2.0) + 0.5,
                                cb_center[0] + (w / 2.0) - 0.5,
                                cb_center[1] + (h / 2.0) - 0.5,
                            ),
                            dtype=np.float32,
                        )
                        current_box = hockey_mom.shift_box_to_edge(current_box)
                    if self._args.plot_camera_tracking:
                        vis.plot_rectangle(
                            online_im,
                            current_box,
                            color=(60, 60, 60),  # Gray
                            thickness=6,
                            label="after-aspect",
                        )

                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

                def _fix_aspect_ratio(box):
                    box = hockey_mom.make_box_proper_aspect_ratio(
                        frame_id=self._frame_id,
                        the_box=box,
                        desired_aspect_ratio=self._final_aspect_ratio,
                        max_in_aspec_ratio=False,
                        no_max_in_aspec_ratio_at_edges=False,
                    )
                    return hockey_mom.shift_box_to_edge(box)

                stuck = hockey_mom.did_direction_change(dx=True, dy=False, reset=False)

                if self._args.max_in_aspec_ratio:
                    if last_sticky_temporal_box is not None:
                        gaussian_factor = hockey_mom.get_gaussian_y_from_image_x_position(center(last_sticky_temporal_box)[0])
                    else:
                        gaussian_factor = 1
                    gaussian_mult = 10
                    gaussian_add = gaussian_factor * gaussian_mult
                    #print(f"gaussian_factor={gaussian_factor}, gaussian_add={gaussian_add}")
                    sticky_size = hockey_mom._camera_box_max_speed_x * (6 + gaussian_add)
                    unsticky_size = sticky_size * 3 / 4
                    #movement_speed_divisor = 1.0
                else:
                    sticky_size = hockey_mom._camera_box_max_speed_x * 5
                    unsticky_size = sticky_size / 2
                    #movement_speed_divisor = 3.0

                if last_sticky_temporal_box is not None:
                    if self._args.plot_sticky_camera:
                        vis.plot_rectangle(
                            online_im,
                            last_sticky_temporal_box,
                            color=(255, 255, 255),
                            thickness=6,
                        )
                        # sticky circle
                        #cc = center(current_box)
                        cc = center(fine_tracking_box)
                        cl = center(last_sticky_temporal_box)

                        def _to_int(vals):
                            return [int(i) for i in vals]

                        cv2.circle(
                            online_im,
                            _to_int(cl),
                            radius=int(sticky_size),
                            color=(255, 0, 0) if stuck else (255, 255, 255),
                            thickness=3,
                        )
                        cv2.circle(
                            online_im,
                            _to_int(cl),
                            radius=int(unsticky_size),
                            color=(0, 255, 255) if stuck else (128, 128, 255),
                            thickness=2,
                        )
                        vis.plot_point(online_im, cl, color=(0, 0, 255), thickness=10)
                        vis.plot_point(online_im, cc, color=(0, 255, 0), thickness=6)
                        vis.plot_line(
                            online_im, cl, cc, color=(255, 255, 255), thickness=2
                        )

                        # current velocity vector
                        vis.plot_line(
                            online_im,
                            cl,
                            [
                                cl[0] + hockey_mom._current_camera_box_speed_x,
                                cl[1] + hockey_mom._current_camera_box_speed_y,
                            ],
                            color=(0, 0, 0),
                            thickness=2,
                        )

                #cdist = center_distance(current_box, last_sticky_temporal_box)
                cdist = center_x_distance(current_box, last_sticky_temporal_box)

                # if stuck and (center_distance(current_box, last_sticky_temporal_box) > 30 or hockey_mom.is_fast(speed=10)):
                if stuck and (
                    # center_distance(current_box, last_sticky_temporal_box) > 30
                    # Past some distance of number of frames at max speed
                    cdist
                    > sticky_size
                ):
                    hockey_mom.control_speed(
                        hockey_mom._camera_box_max_speed_x / 6,
                        hockey_mom._camera_box_max_speed_y / 6,
                        #set_speed_x=True,
                        set_speed_x=False,
                    )
                    hockey_mom.did_direction_change(dx=True, dy=True, reset=True)
                    stuck = False
                elif cdist < unsticky_size:
                    stuck = hockey_mom.set_direction_changed(dx=True, dy=True)

                if not stuck:
                    #xx0 = center(current_box)[0]
                    current_box, last_sticky_temporal_box = _apply_temporal(
                        current_box,
                        last_sticky_temporal_box, scale_speed=1.0,
                        verbose=True,
                    )
                    #xx1 = center(current_box)[0]
                    #print(f'A final temporal x change: {xx1 - xx0}')
                    current_box = _fix_aspect_ratio(current_box)
                    #xx1 = center(current_box)[0]
                    #print(f'final temporal x change: {xx1 - xx0}')
                    # vis.plot_rectangle(
                    #     online_im,
                    #     current_box,
                    #     color=(0, 128, 255),
                    #     thickness=15,
                    #     label="edge-scaled",
                    # )
                    assert np.isclose(
                        aspect_ratio(current_box), self._final_aspect_ratio
                    )
                    hockey_mom.did_direction_change(dx=True, dy=True, reset=True)
                elif last_sticky_temporal_box is None:
                    last_sticky_temporal_box = current_box.copy()
                    assert np.isclose(
                        aspect_ratio(current_box), self._final_aspect_ratio
                    )
                else:
                    current_box = last_sticky_temporal_box.copy()
                    current_box = _fix_aspect_ratio(current_box)
                    assert np.isclose(
                        aspect_ratio(current_box), self._final_aspect_ratio
                    )

                if (
                    self._args.apply_fixed_edge_scaling
                    and self._args.apply_fixed_edge_scaling
                ):
                    current_box = hockey_mom.apply_fixed_edge_scaling(
                        current_box,
                        edge_scaling_factor=self._args.fixed_edge_scaling_factor,
                    )
                    if self._args.plot_camera_tracking:
                        vis.plot_rectangle(
                            online_im,
                            current_box,
                            color=(255, 0, 255),
                            thickness=5,
                            label="edge-scaled",
                        )

                if stuck and self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(0, 160, 255),
                        thickness=10,
                        label="stuck",
                    )
                elif self._args.plot_camera_tracking:
                    vis.plot_rectangle(
                        online_im,
                        current_box,
                        color=(160, 160, 255),
                        thickness=6,
                        label="post-sticky",
                    )

                # current_box = hockey_mom.make_box_proper_aspect_ratio(
                #     frame_id=self._frame_id,
                #     the_box=current_box,
                #     desired_aspect_ratio=self._final_aspect_ratio,
                #     max_in_aspec_ratio=False, # FALSE HERE HARD CODED
                #     no_max_in_aspec_ratio_at_edges=self._args.no_max_in_aspec_ratio_at_edges,
                # )
                # assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

                # Plot the trajectories
                if self._args.plot_individual_player_tracking:
                    online_im = vis.plot_trajectory(
                        online_im, hockey_mom.get_image_tracking(online_ids), online_ids
                    )

                # kmeans = KMeans(n_clusters=3)
                # kmeans.fit(hockey_mom.online_image_center_points)
                # plt.scatter(x, y, c=kmeans.labels_)
                # plt.show()
            assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
            imgproc_data = ImageProcData(
                frame_id=self._frame_id,
                img=online_im,
                current_box=current_box,
                save_dir=save_dir,
            )
            timer.toc()
            # Only let it get ahead around 25 frames so as not to use too much
            # memory for no gain
            while self._imgproc_queue.qsize() > 25:
                time.sleep(0.001)
            self._imgproc_queue.put(imgproc_data)
