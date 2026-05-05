filegroup(
    name = "libtorch_so_files",
    srcs = [
%{LIBTORCH_SO_FILES}
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtorch",
    srcs = [":libtorch_so_files"],
    hdrs = glob(["include/**/*.h*"]),
    defines = [
        # Must match `torch._C._GLIBCXX_USE_CXX11_ABI` for the installed torch.
        "_GLIBCXX_USE_CXX11_ABI=%{GLIBCXX_USE_CXX11_ABI}",
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
        "_GLIBCXX_USE_CXX11_ABI=%{GLIBCXX_USE_CXX11_ABI}",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libtorch",
        "@local_config_python//:python_headers",
    ],
)
