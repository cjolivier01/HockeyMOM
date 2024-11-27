import argparse
import sys
import traceback
from typing import Any

import cv2
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, GObject, Gst


class VideoTranscoder:
    def __init__(self, input_file: str, output_file: str, bitrate: int = 2000):
        self.input_file = input_file
        self.output_file = output_file
        self.bitrate = bitrate
        Gst.init(None)
        self.loop = GObject.MainLoop()
        self.pipeline = self.setup_pipeline()

    def setup_pipeline(self) -> Gst.Pipeline:
        pipeline = Gst.parse_launch(
            f"filesrc location={self.input_file} ! "
            "h265parse ! "
            "avdec_h265 ! "  # Software-based decoder for HEVC
            "videoconvert ! "  # Convert video formats to raw for further processing
            f"x264enc bitrate={self.bitrate} tune=zerolatency ! "
            "qtmux ! "
            f"filesink location={self.output_file}"
        )        
        return pipeline

    def on_message(self, bus: Gst.Bus, message: Gst.Message, loop: GObject.MainLoop) -> bool:
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True

    def run(self) -> None:
        bus = self.pipeline.get_bus()
        bus.add_watch(GLib.PRIORITY_DEFAULT, self.on_message, self.loop)
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            traceback.print_exc()
        finally:
            self.pipeline.set_state(Gst.State.NULL)


def check_opencv_gstreamer_support():
    # Get build information from OpenCV
    build_info = cv2.getBuildInformation()

    # Check if GStreamer is mentioned in the build information
    if 'GStreamer:' in build_info:
        start = build_info.index('OpenCV GStreamer:')
        end = build_info.index('\n', start)
        gstreamer_info = build_info[start:end]
        print(gstreamer_info)
    else:
        print("GStreamer is not mentioned in OpenCV's build information.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcode a video file with specific bitrate.")
    parser.add_argument("input_file", help="Input video file path")
    parser.add_argument("output_file", help="Output video file path")
    parser.add_argument("--bitrate", type=int, default=2000, help="Bitrate for output video in kbps")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    transcoder = VideoTranscoder(args.input_file, args.output_file, args.bitrate)
    transcoder.run()
