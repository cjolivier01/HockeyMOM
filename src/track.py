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

from typing import Dict, List

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


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

def _make_pruned_map(map, allowed_keys):
    key_set = set(allowed_keys)
    new_map = dict()
    for map_key in map.keys():
        if map_key in key_set:
            new_map[map_key] = map[map_key]
    return new_map

def _normalize_map(map, reference_map):
    for key in reference_map:
        if key not in map:
            map[key] = reference_map[key]
    return map


class VideoFrame(object):

    def __init__(self, image_width:int, image_height:int, scale_width=0.1, scale_height=0.05):
        self.image_width_ = image_width
        self.image_height_ = image_height
        self.vertical_center_ = image_width / 2
        self.horizontal_center_ = image_height / 2
        self.scale_width_ = scale_width
        self.scale_height_ = scale_height

    def point_ratio_vertical_away(self, point):
        """
        Vertical is farther away towards the top
        """
        # Is the first one X?
        y = point[0]
        return y / self.image_height_

    def point_ratio_horizontal_away(self, point):
        """
        Horizontal is farther away towards the left anf right sides
        """
        # Is the first one X?
        x = point[1]
        dx = abs(x - self.horizontal_center_)
        # TODO: Probably can work this out with trig, the actual linear distance,
        # # esp since we know the rink's size
        # Just do a meatball calculation for now...
        return dx / self.horizontal_center_


class PositionHistory(object):
    def __init__(self, id: int, video_frame: VideoFrame):
        self.id_ = id
        self.video_frame_ = video_frame
        self.position_history_ = list()

    @property
    def id(self):
        return self.id_

    @property
    def position_history(self):
        return self.position_history_

    @staticmethod
    def center_point(tlwh):
        top = tlwh[0]
        left = tlwh[1]
        width = tlwh[2]
        height = tlwh[3]
        x_center = left + width / 2
        y_center = top + height / 2
        return np.array((x_center, y_center), dtype=np.float32)

    def speed(self):
        if len(self.position_history) < 2:
            return 0.0
        speed_sum = 0
        movement_count = len(self.position_history_) - 1
        for i in range(movement_count):
          start_point = self.center_point(self.position_history_[i])
          end_point = self.center_point(self.position_history_[i+1])
          center_point_y = (end_point[0] + start_point[0]) / 2
          center_point_x = (end_point[1] + start_point[1]) / 2
          center_point = np.array((center_point_y, center_point_x))
          ratio_y = self.video_frame_.point_ratio_vertical_away(center_point)
          ratio_x = self.video_frame_.point_ratio_horizontal_away(center_point)
          dy = end_point[0] - start_point[0]
          dx = end_point[1] - start_point[1]
          dy += dy * ratio_y
          dx += dx * ratio_x
          #velocity_vector = np.array()
          #distance_traveled = np.linalg.norm(velocity_vector)
          distance_traveled = math.sqrt(dx * dx + dy * dy)
          speed = distance_traveled / len(self.position_history)
          speed_sum += speed
        average_speed = speed_sum / movement_count
        return average_speed



def _get_id_to_pos_history_map(tlwhs_history: Dict, speed_history: int, online_ids: List[int], video_frame: VideoFrame):
    assert len(tlwhs_history) > 0
    online_ids_set = set(online_ids)
    id_to_pos_history = dict()
    for id_to_tlwhs_map in reversed(tlwhs_history):
        # Iterating histories in reverse order
        for this_id in id_to_tlwhs_map.keys():
            if this_id not in online_ids_set:
                continue
            if this_id not in id_to_pos_history:
                pos_hist = PositionHistory(this_id, video_frame)
                id_to_pos_history[this_id] = pos_hist
            else:
                pos_hist = id_to_pos_history[this_id]
            if speed_history <= 0 or len(pos_hist.position_history) < speed_history:
                tlwh = id_to_tlwhs_map[this_id]
                pos_hist.position_history.append(tlwh)
    return id_to_pos_history


def _get_id_to_speed_map(id_to_pos_history: Dict[int, PositionHistory]):
    id_to_speed_map = dict()
    for id in id_to_pos_history.keys():
        id_to_speed_map[id] = id_to_pos_history[id].speed()
    return id_to_speed_map


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    online_tlwhs_history = list()
    max_history = 26
    speed_history = 26
    video_frame = None
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
        
        if video_frame is None:    
            video_frame = VideoFrame(
              image_width=img.shape[2], image_height=img.shape[1],
              scale_width=0.1, scale_height=0.05
            )
            
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_tlwhs_map = dict()
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_tlwhs_map[tid] = tlwh
                #online_scores.append(t.score)
            else:
                print(f'Box area too small (< {opt.min_box_area}): {tlwh[2] * tlwh[3]} or vertical (vertical={vertical})')
        timer.toc()

        online_tlwhs_history.append(online_tlwhs_map)
        if len(online_tlwhs_history) > max_history:
            online_tlwhs_history = online_tlwhs_history[len(online_tlwhs_history) - max_history:]

        id_to_tlwhs_history_map = _get_id_to_pos_history_map(online_tlwhs_history, speed_history, online_ids, video_frame)

        id_to_speed_map = _get_id_to_speed_map(id_to_tlwhs_history_map)
        #id_to_speed_map = _make_pruned_map(id_to_speed_map, online_ids)
        assert len(id_to_speed_map) == len(online_ids)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            #online_im = img0

            these_online_speeds = []
            for id in online_ids:
                these_online_speeds.append(id_to_speed_map[id])

            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids,
                                          frame_id=frame_id, fps=1. / timer.average_time,
                                          speeds=these_online_speeds)
            current_pos_map = None
            for pos_map in reversed(online_tlwhs_history):
                if current_pos_map is None:
                    current_pos_map = pos_map
                else:
                  # Prune any items not currently online
                  pos_map = _make_pruned_map(pos_map, online_ids)
                  pos_map = _normalize_map(pos_map, current_pos_map)
                  assert len(pos_map) == len(current_pos_map)
                # Make order the same
                these_online_tlws = []
                for id in online_ids:
                    these_online_tlws.append(pos_map[id])
                online_im = vis.plot_trajectory(online_im, these_online_tlws, online_ids)
        if show_image:
            # sleep(3)
            if frame_id % 1 == 0:
              cv2.imshow('online_im', online_im)
              cv2.waitKey(1)
            pass
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
