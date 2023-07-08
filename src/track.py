from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import math
import time
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import typing
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#from PIL import Image

from typing import Dict, List

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

from camera.camera import HockeyMOM

def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #online_tlwhs_history = list()
    # max_history = 26
    # speed_history = 26
    plot_tracking = True
    hockey_mom = None
    current_bounding_box = None
    last_bounding_box = None
    last_fast_bounding_box = None
    plot_interias = False
    #image_viewer: Viewer = None
    camera_box = None
    show_image_interval = 1
    last_both_boxes = None
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        if hockey_mom is None:
            hockey_mom = HockeyMOM(
              image_width=img.shape[2], image_height=img.shape[1],
              scale_width=0, scale_height=0
            )

        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
            else:
                print(f'Box area too small (< {opt.min_box_area}): {tlwh[2] * tlwh[3]} or vertical (vertical={vertical})')
        timer.toc()

        # online_tlwhs = []
        # online_ids = []

        hockey_mom.append_online_objects(online_ids, online_tlwhs)

        hockey_mom.calculate_clusters(n_clusters=2)

        largest_cluster_id_set = hockey_mom.get_largest_cluster_id_set()

        #last_fast_online_ids = None

        # if last_bounding_box is None:
        #     last_bounding_box = hockey_mom.get_current_bounding_box()

        #id_to_speed_map = _make_pruned_map(id_to_speed_map, online_ids)
        #assert len(id_to_speed_map) == len(online_ids)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = img0
            fast_bounding_box = None
            # if plot_tracking:
            #     online_im = vis.plot_tracking(online_im, online_tlwhs, online_ids, frame_id=frame_id, fps=1./timer.average_time)

            if i == 0:
                # First step, we have no velocities, so pick everyone
                # fast_online_ids = online_ids
                # fast_online_tlwhs = online_tlwhs
                # fast_online_speeds = np.zeros(len(online_ids))
                # fast_ids = online_ids
                # fast_online_tlwhs = online_tlwhs
                fast_ids = []
                fast_online_tlwhs = []
            else:
                # fast_online_ids = []
                # fast_online_tlwhs = []
                # fast_online_speeds = []
                fast_ids = hockey_mom.get_fast_ids()
                if fast_ids:
                    fast_online_tlwhs = [hockey_mom.get_tlwh(id) for id in fast_ids]
                    fast_bounding_box = hockey_mom.get_current_bounding_box(ids=fast_ids)
                else:
                    #fast_ids = online_ids
                    #fast_online_tlwhs = online_tlwhs
                    pass
                # for fast_id, fast_tlwhs in zip(online_ids, online_tlwhs):
                #     if fast_id in fast_ids:
                #         fast_online_ids.append(fast_id)
                #         fast_online_tlwhs.append(fast_tlwhs)
                #         fast_online_speeds.append(hockey_mom.get_speed(fast_id))

            #last_bounding_box = hockey_mom.get_current_bounding_box(obj_ids=fast_ids)

            if fast_ids and False:
                online_im = vis.plot_tracking(online_im, fast_online_tlwhs, fast_ids,
                                              frame_id=frame_id, fps=1. / timer.average_time)
                # if current_bounding_box is None:
                #     current_bounding_box = last_bounding_box
                # else:
                #     current_bounding_box = hockey_mom.translate_box(
                #       current_bounding_box,
                #       last_bounding_box,
                #       max_x=30,
                #       max_y=30,
                #       clamp_box=hockey_mom.clamp_box)
                #     print(current_bounding_box)
            else:
                these_online_speeds = []
                for id in online_ids:
                    these_online_speeds.append(hockey_mom.get_spatial_speed(id))
                online_im = vis.plot_tracking(online_im, online_tlwhs, online_ids,
                                              frame_id=frame_id, fps=1. / timer.average_time,
                                              speeds=these_online_speeds)
                pass
            if fast_ids:
                current_bounding_box = hockey_mom.get_current_bounding_box(ids=fast_ids)
                last_bounding_box = current_bounding_box
            else:
                current_bounding_box = last_bounding_box

            if plot_tracking:
                # if fast_bounding_box is not None:
                #     cv2.rectangle(online_im, fast_bounding_box[0:2], fast_bounding_box[2:4], color=(255, 255, 0), thickness=6)
                #     last_fast_bounding_box = fast_bounding_box
                # elif last_fast_bounding_box is not None:
                #     cv2.rectangle(online_im, last_fast_bounding_box[0:2], last_fast_bounding_box[2:4], color=(128, 128, 0), thickness=6)

                largest_cluster_ids = hockey_mom.prune_not_in_largest_cluster(online_ids)
                #largest_cluster_ids = hockey_mom.prune_not_in_largest_cluster(fast_ids)
                if largest_cluster_ids:
                    largest_cluster_ids_box = hockey_mom.get_current_bounding_box(largest_cluster_ids)
                    largest_cluster_ids_box = hockey_mom.make_normalized_bounding_box(largest_cluster_ids_box)
                    cv2.rectangle(online_im, largest_cluster_ids_box[0:2], largest_cluster_ids_box[2:4], color=(0, 255, 128), thickness=4)

            if plot_tracking and current_bounding_box is not None:
                cv2.rectangle(online_im, current_bounding_box[0:2], current_bounding_box[2:4], color=(255, 0, 0), thickness=4)
                # Union box

            if current_bounding_box is not None and largest_cluster_ids_box is not None:
                both_boxes = HockeyMOM._union(current_bounding_box, largest_cluster_ids_box)
            elif largest_cluster_ids_box is not None:
                both_boxes = largest_cluster_ids_box
            elif current_bounding_box is not None:
                both_boxes = current_bounding_box

            cv2.rectangle(online_im, both_boxes[0:2], both_boxes[2:4], color=(255, 0, 255), thickness=8)
            if last_both_boxes is not None:
                both_boxes = hockey_mom.translate_box(last_both_boxes, both_boxes, clamp_box=hockey_mom._video_frame.box())
            cv2.rectangle(online_im, both_boxes[0:2], both_boxes[2:4], color=(0, 0, 255), thickness=8)
            last_both_boxes = both_boxes


            # if not fast_online_ids:
            #     fast_online_ids = last_fast_online_ids
            # elif fast_online_ids:
            #     last_fast_online_ids = fast_online_ids
            # if fast_online_ids:
            #     online_im = vis.plot_camera(image=online_im, obj_ids=fast_online_ids, hockey_mom=hockey_mom)

            # TODO: slowly towards last_bounding_box
            #current_bounding_box = last_bounding_box

            # current_pos_map = None
            # for pos_map in reversed(online_tlwhs_history):
            #     if current_pos_map is None:
            #         current_pos_map = pos_map
            #     else:
            #       # Prune any items not currently online
            #       pos_map = hockey_mom._make_pruned_map(pos_map, online_ids)
            #       pos_map = hockey_mom._normalize_map(pos_map, current_pos_map)
            #       assert len(pos_map) == len(current_pos_map)

            # # Make order the same
            # these_online_tlws = []
            # for id in online_ids:
            #     these_online_tlws.append(pos_map[id])
            # # Plot the trajectories
            if plot_tracking:
                online_im = vis.plot_trajectory(online_im, hockey_mom.get_image_tracking(online_ids), online_ids)

            # kmeans = KMeans(n_clusters=3)
            # kmeans.fit(hockey_mom.online_image_center_points)
            # plt.scatter(x, y, c=kmeans.labels_)
            # plt.show()

        if show_image:
            # sleep(3)
            if frame_id % show_image_interval == 0:
              cv2.imshow('online_im', online_im)
              cv2.waitKey(1)
              pass
            pass

        if plot_interias:
            vis.plot_kmeans_intertias(hockey_mom=hockey_mom)

        if save_dir is not None:
            #cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            cv2.imwrite(os.path.join(save_dir, '{:05d}.png'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
