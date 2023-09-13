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
import uuid

logger.setLevel(logging.INFO)

ORIGINAL_IMAGE_SIZE = (1088, 608)
ORIGINAL_PROCESS_IMAGE_SIZE = (1920, 1080)

def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')


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

    dataloader = datasets.LoadVideoWithOrig(path=opt.input_video, img_size=opt.img_size, process_img_size=opt.process_img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
    print(f"Video frame rate: {frame_rate}")

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'hockey-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.png -b:v 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        # trunk-ignore(bandit/B605)
        os.system(cmd_str)

        output_video_with_audio_path = osp.join(result_root, 'final-hockey-results.mp4')
        copy_audio(opt.input_video, output_video_path, output_video_with_audio_path)


def copy_audio(original_video: str, soundless_video: str, final_audio_video: str):
    # output audio from original
    output_audio_path = f"/tmp/output-audio-{uuid.uuid4().hex}.mp3"
    cmd_str = f'ffmpeg -i {original_video} -q:a 0 -map a {output_audio_path}'
    print(cmd_str)
    os.system(cmd_str)
    # attach audio to new video
    cmd_str = f'ffmpeg -i {soundless_video} -i {output_audio_path} -map 0:v -map 1:a -c:v copy -shortest {final_audio_video}'
    print(cmd_str)
    os.system(cmd_str)
    # delete temp audio
    if os.path.isfile(output_audio_path):
        os.unlink(output_audio_path)


if __name__ == '__main__':
    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
