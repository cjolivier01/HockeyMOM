"""GStreamer-based transcoding experiments.

This module is currently experimental and not used in the main pipeline,
but demonstrates how to wire GStreamer pipelines for video processing.
"""

import argparse
import traceback
from typing import Optional, Tuple

import cv2
import gi
from gi.repository import GLib, GObject, Gst  # noqa: E402

gi.require_version("Gst", "1.0")


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
            "playbin uri=file:///olivier-pool/Videos/ev-bs-short/GX010085.MP4",
            # f"filesrc location={self.input_file} ! "
            # "h265parse ! "
            # "avdec_h265 ! "  # Software-based decoder for HEVC
            # "videoconvert ! "  # Convert video formats to raw for further processing
            # f"x264enc bitrate={self.bitrate} tune=zerolatency ! "
            # "qtmux ! "
            # f"filesink location={self.output_file}"
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


class GStreamerVideoCapture:
    def __init__(self, source: str) -> None:
        Gst.init(None)
        self.pipeline = None
        self.source = source
        self.is_opened = False
        self.appsink = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        if isinstance(self.source, int):
            pipeline_description = f"v4l2src device=/dev/video{self.source} ! videoconvert ! appsink name=appsink"
        elif isinstance(self.source, str):
            pipeline_description = f"filesrc location={self.source} ! decodebin ! nvvideoconvert ! appsink name=appsink"
        else:
            raise ValueError("Invalid source type. Must be int (camera index) or str (file path).")

        self.pipeline = Gst.parse_launch(pipeline_description)
        self.appsink = self.pipeline.get_by_name("appsink")
        if self.appsink is None:
            raise RuntimeError("Failed to create appsink element in the pipeline.")
        self.is_opened = True

    def isOpened(self) -> bool:
        return self.is_opened

    def read(self) -> Tuple[bool, Optional[bytes]]:
        if not self.is_opened:
            return False, None

        self.pipeline.set_state(Gst.State.PLAYING)
        sample = self.appsink.emit("pull-sample")
        if sample is None:
            return False, None

        buf = sample.get_buffer()
        result, map_info = buf.map(Gst.MapFlags.READ)
        if not result:
            return False, None

        frame_data = map_info.data.tobytes()
        buf.unmap(map_info)
        return True, frame_data

    def release(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_opened = False

    def set(self, prop_id: int, value: float) -> bool:
        # GStreamer can set some properties, but it's not a direct 1:1 with OpenCV
        # We would need to handle GStreamer properties explicitly here.
        # For simplicity, we will return False for now
        return False

    def get(self, prop_id: int) -> float:
        # Similar to `set`, GStreamer does not provide direct 1:1 properties with OpenCV
        # Handle this appropriately by returning default or not implemented values
        return 0.0


def check_opencv_gstreamer_support():
    # Get build information from OpenCV
    build_info = cv2.getBuildInformation()

    # Check if GStreamer is mentioned in the build information
    if "GStreamer:" in build_info:
        start = build_info.index("OpenCV GStreamer:")
        end = build_info.index("\n", start)
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
    # transcoder = VideoTranscoder(args.input_file, args.output_file, args.bitrate)
    # transcoder.run()
    gst_cap = GStreamerVideoCapture(args.input_file)
    if gst_cap.isOpened():
        ret, frame = gst_cap.read()
        if ret:
            print(f"Read frame of shape: {frame.shape}")
        else:
            print("Failed to read frame")

    gst_cap.release()
