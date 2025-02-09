filegroup(
    name = "libtorch_so_files",
    srcs = [
        "lib/libc10_cuda.so",
        "lib/libtorch_cuda.so",
        "lib/libc10.so",
        "lib/libshm.so",
        "lib/libtorch.so",
        "lib/libtorch_cpu.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtorch",
    srcs = [":libtorch_so_files"],
    hdrs = glob(["include/**/*.h*"]),
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    visibility = ["//visibility:public"],
    linkstatic = 1,
)

filegroup(
    name = "libtorch_python_so_files",
    srcs = [
        "lib/libtorch_python.so",
        ":libtorch_so_files",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtorch_python",
    srcs = ["lib/libtorch_python.so"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtorch",
        "@local_config_python//:python_headers",
    ],
)
