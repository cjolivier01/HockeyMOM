load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "kmeans",
    srcs = [
      "kmeans.cpp",
      "omp_kmeans.cpp",
      "seq_kmeans.cpp",
      "wtime.cpp",
    ],
    hdrs = glob(["*.h"]),
    copts = [
        "-fopenmp",
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
