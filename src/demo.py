from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)

ORIGINAL_IMAGE_SIZE = (1088, 608)
ORIGINAL_PROCESS_IMAGE_SIZE = (1920, 1080)

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    #opt.img_size = (4096, 1800)
    opt.img_size = (4096, 1024)
    #opt.img_size = (4095, 1800)
    #opt.img_size = (1088, 608)
    #opt.img_size = ORIGINAL_IMAGE_SIZE
    #opt.img_size = (2194, 1214)
    
    opt.process_img_size = opt.img_size
    #opt.process_img_size = ORIGINAL_PROCESS_IMAGE_SIZE

    dataloader = datasets.LoadVideo(path=opt.input_video, img_size=opt.img_size, process_img_size=opt.process_img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=True, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        #cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.png -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
