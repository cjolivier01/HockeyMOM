# Bazel is only available for amd64 and arm64.
# load("//:buildfiles/third_party/opencv.bzl", "get_opencv_version")

config_setting(
    name = "aarch64-linux-gnu",
    constraint_values = ["@platforms//cpu:aarch64"],
)

config_setting(
    name = "x86_64-linux-gnu",
    constraint_values = ["@platforms//cpu:x86_64"],
)

# OPENCV_VERSION = "opencv5"
OPENCV_VERSION = "opencv4"

cc_library(
    name = "opencv",
    hdrs = glob([
        # "opencv5/opencv2/**/*.h*",
    ]) + select({
        ":aarch64-linux-gnu": [
            # "aarch64-linux-gnu/opencv5/opencv2/cvconfig.h"
        ] + glob([
            "opencv4/opencv2/**/*.h*",
        ]),
        ":x86_64-linux-gnu": [
            # "x86_64-linux-gnu/opencv5/opencv2/cvconfig.h",
        ] + glob([
            OPENCV_VERSION + "/opencv2/**/*.h*",
        ]),
        "//conditions:default": [],
    }),
    includes = [
    ] + select({
        ":aarch64-linux-gnu": [
            "opencv4",
            "x86_64-linux-gnu/opencv4/opencv2",
            # "aarch64-linux-gnu/opencv5",
        ],
        ":x86_64-linux-gnu": [
            OPENCV_VERSION,
            "x86_64-linux-gnu/" + OPENCV_VERSION,
            "x86_64-linux-gnu/" + OPENCV_VERSION + "/opencv2",
        ],
        "//conditions:default": [],
    }),
    linkopts = [
        # "-L/usr/local/lib",
        "-l:libopencv_core.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
    ],
    visibility = ["//visibility:public"],
)
