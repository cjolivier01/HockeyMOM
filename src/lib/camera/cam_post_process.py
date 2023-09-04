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

from .camera import aspect_ratio, width, height

from hockeymom import core


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
class DefaultArguments(argparse.Namespace):
    def __init__(self, args: argparse.Namespace = None):
        # Display the image every frame (slow)
        self.show_image = True

        # Draw individual player boxes, tracking ids, speed and history trails
        self.plot_individual_player_tracking = False

        # Draw intermediate boxes which are used to compute the final camera box
        self.plot_camera_tracking = False

        # Use a differenmt algorithm when fitting to the proper aspect ratio,
        # such that the box calculated is much larger and often takes
        # the entire height.  The drawback is there's not much zooming.
        self.max_in_aspec_ratio = False

        # Skip some number of frames before post-processing. Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.skip_frame_count = 0

        # Stop at the given frame and (presumably) output the final video.
        # Useful for debugging a
        # particular section of video and being able to reach
        # that portiuon of the video more quickly
        self.stop_at_frame = None

        # Make the image the same relative dimensions as the initial image,
        # such that the highest possible resolution is available when the camera
        # box is either the same height or width as the original video image
        # (Slower, but better final quality)
        self.scale_to_original_image = False

        # Crop the final image to the camera window (possibly zoomed)
        self.crop_output_image = True

        # Don't crop image, but performa of the calculations
        # except for the actual image manipulations
        self.fake_crop_output_image = False

        # Use cuda for final image resizing (if possible)
        self.use_cuda = False


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
        self._opt = opt
        self._thread = None
        self._imgproc_thread = None
        self._use_fork = use_fork
        self._final_aspect_ratio = 16.0 / 9.0

    def start(self):
        if self._use_fork:
            self._child_pid = os.fork()
            if not self._child_pid:
                self.device = torch.device("cuda:0")
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
        plot_interias = False
        show_image_interval = 1
        skip_frames_before_show = 0
        timer = Timer()
        while True:
            imgproc_data = self._imgproc_queue.get()
            if imgproc_data is None:
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
            if self._args.crop_output_image:
                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)
                # print(f"crop ar={aspect_ratio(current_box)}")
                intbox = [int(i) for i in current_box]
                x1 = intbox[0]
                # x2 = intbox[2]
                y1 = intbox[1]
                y2 = intbox[3]
                x2 = x1 + int(float(y2 - y1) * self._final_aspect_ratio)

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

                if self._args.plot_individual_player_tracking:
                    # Plot the player boxes
                    online_im = vis.plot_tracking(
                        online_im,
                        online_tlwhs,
                        online_ids,
                        frame_id=self._frame_id,
                        fps=1.0 / timer.average_time if timer.average_time else 1000.0,
                        speeds=these_online_speeds,
                    )

                # Examine as 2 clusters
                largest_cluster_ids_2 = hockey_mom.prune_not_in_largest_cluster(
                    n_clusters=2, ids=online_ids
                )
                if largest_cluster_ids_2:
                    largest_cluster_ids_box2 = hockey_mom.get_current_bounding_box(
                        largest_cluster_ids_2
                    )
                    if self._args.plot_camera_tracking:
                        vis.plot_rectangle(
                            online_im,
                            largest_cluster_ids_box2,
                            color=(128, 0, 0),  # dark red
                            thickness=6,
                            label="largest_cluster_ids_box2",
                        )
                else:
                    largest_cluster_ids_box2 = None

                # Examine as 3 clusters
                largest_cluster_ids_3 = hockey_mom.prune_not_in_largest_cluster(
                    n_clusters=3, ids=online_ids
                )
                if largest_cluster_ids_3:
                    largest_cluster_ids_box3 = hockey_mom.get_current_bounding_box(
                        largest_cluster_ids_3
                    )
                    if self._args.plot_camera_tracking:
                        vis.plot_rectangle(
                            online_im,
                            largest_cluster_ids_box3,
                            color=(0, 0, 128),  # dark blue
                            thickness=6,
                            label="largest_cluster_ids_box3",
                        )
                else:
                    largest_cluster_ids_box3 = None

                if (
                    largest_cluster_ids_2 is not None
                    and largest_cluster_ids_box3 is not None
                ):
                    current_box = hockey_mom.union_box(
                        largest_cluster_ids_box2, largest_cluster_ids_box3
                    )
                elif largest_cluster_ids_2 is not None:
                    current_box = largest_cluster_ids_box2
                elif largest_cluster_ids_3 is not None:
                    current_box = largest_cluster_ids_box3
                else:
                    current_box = hockey_mom._video_frame.box()

                # current_box = hockey_mom.ratioed_expand(current_box)

                outside_expanded_box = current_box.copy()

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

                def _apply_temporal():
                    #
                    # Temporal: Apply velocity and acceleration
                    #
                    nonlocal current_box, last_temporal_box, self
                    current_box = hockey_mom.get_next_temporal_box(
                        current_box, last_temporal_box
                    )
                    last_temporal_box = current_box.copy()

                    if self._args.plot_camera_tracking:
                        # print(f"get_next_temporal_box() ar={aspect_ratio(new_temporal_box)}, box={new_temporal_box}")
                        vis.plot_rectangle(
                            online_im,
                            current_box,
                            color=(128, 255, 128),
                            thickness=2,
                            label="next_temporal_box",
                        )
                    return current_box

                current_box = _apply_temporal()

                hockey_mom.curtail_velocity_if_outside_box(
                    current_box, outside_expanded_box
                )

                #
                # Aspect Ratio
                #
                current_box = hockey_mom.clamp(current_box)
                current_box = hockey_mom.make_box_proper_aspect_ratio(
                    frame_id=self._frame_id,
                    the_box=current_box,
                    desired_aspect_ratio=self._final_aspect_ratio,
                    max_in_aspec_ratio=self._args.max_in_aspec_ratio,
                )
                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

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

                assert np.isclose(aspect_ratio(current_box), self._final_aspect_ratio)

                # Plot the trajectories
                if self._args.plot_camera_tracking:
                    online_im = vis.plot_trajectory(
                        online_im, hockey_mom.get_image_tracking(online_ids), online_ids
                    )

                # kmeans = KMeans(n_clusters=3)
                # kmeans.fit(hockey_mom.online_image_center_points)
                # plt.scatter(x, y, c=kmeans.labels_)
                # plt.show()
            imgproc_data = ImageProcData(
                frame_id=self._frame_id,
                img=online_im,
                current_box=current_box,
                save_dir=save_dir,
            )
            timer.toc()
            # while self._imgproc_queue.qsize() > 0:
            #     time.sleep(0.001)
            self._imgproc_queue.put(imgproc_data)
