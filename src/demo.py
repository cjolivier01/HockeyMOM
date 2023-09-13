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
from lib.ffmpeg import copy_audio

logger.setLevel(logging.INFO)

ORIGINAL_IMAGE_SIZE = (1088, 608)
ORIGINAL_PROCESS_IMAGE_SIZE = (1920, 1080)

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')

    image_number = 0

    #opt.img_size = (4096, 1024)

    opt.img_size = (4096//2, 1024//2)
    #opt.img_size = (4096, 1800)
    #opt.img_size = (1088, 608)
    #opt.img_size = (1088*2, 608*2)
    #opt.img_size = ORIGINAL_IMAGE_SIZE

    #my_process_image_size = (4096//2, 1024//2)

    opt.process_img_size = opt.img_size
    #opt.process_img_size = my_process_image_size
    #opt.process_img_size = ORIGINAL_PROCESS_IMAGE_SIZE

    input_video_files = opt.input_video.split(',')

    if len(input_video_files) == 2:
        dataloader = datasets.LoadStitchedVideoWithOrig(
            left_file=input_video_files[0],
            right_file=input_video_files[1],
            img_size=opt.img_size,
            process_img_size=opt.process_img_size
        )
    else:
        assert len(input_video_files) == 1
        dataloader = datasets.LoadVideoWithOrig(path=input_video_files[0], img_size=opt.img_size, process_img_size=opt.process_img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
    print(f"Video frame rate: {frame_rate}")

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    output_video_path = osp.join(result_root, 'tracking_output.mov')

    if opt.output_format == 'video':
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.png -b:v 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        # trunk-ignore(bandit/B605)
        os.system(cmd_str)

    if opt.output_format in ["video", "live_video"]:
        output_video_with_audio_path = osp.join(result_root, 'tracking_output-with-audio.mov')
        copy_audio(input_video_files[0], output_video_path, output_video_with_audio_path)


if __name__ == '__main__':
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    opt.output_format = "live_video"
    demo(opt)
