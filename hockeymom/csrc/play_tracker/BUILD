load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "play_tracker",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps=[
      "//hockeymom/csrc/kmeans:kmeans",
    ]
)
