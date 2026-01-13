filegroup(
    name = "libtorch_so_files",
    srcs = [
        "lib/libcaffe2_nvrtc.so",
        "lib/libc10_cuda.so",
        "lib/libtorch_cuda.so",
        "lib/libtorch_cuda_linalg.so",
        "lib/libtorch_global_deps.so",
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
    defines = [
        # Must match `torch._C._GLIBCXX_USE_CXX11_ABI` for the installed torch.
        # Newer CUDA wheels (e.g. cu128) use the C++11 ABI (1).
        "_GLIBCXX_USE_CXX11_ABI=1",
    ],
    includes = [
        "include",
        "include/torch/csrc/api/include",
    ],
    visibility = ["//visibility:public"],
    linkopts = [
      "-Wl,-rpath,lib",
    ],
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
    defines = [
        "_GLIBCXX_USE_CXX11_ABI=1",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libtorch",
        "@local_config_python//:python_headers",
    ],
)
