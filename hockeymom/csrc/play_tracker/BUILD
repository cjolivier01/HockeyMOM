load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "play_tracker",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    copts = [
        "-fopenmp",
        "-fPIC",
        "-Wno-error=comment",
    ],
    deps=[
      "//hockeymom/csrc/kmeans:kmeans",
    ]
)
