load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "kmeans",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    copts = [
        "-fopenmp",
        "-Wall",
        "-std=c++17",
        "-fPIC",
        "-Wno-error=deprecated-declarations",
        "-DWITH_CUDA",
        "-Wno-error=comment",
    ],
    linkopts = [
        "-lgomp",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
