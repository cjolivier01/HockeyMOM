filegroup(
    name = "libpython_files",
    srcs = [
      # "lib/libpython3.11.so"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libpython",
    srcs = [":libpython_files"],
    linkopts = [
        "-Wl,-rpath,lib",
        "-lpthread",
        "-lrt",
        "-ldl",
        "-lutil",
        "-lcrypt",
        "-lm",
    ],
    visibility = ["//visibility:public"],
    deps = ["@local_config_python//:python_headers"],
    linkstatic = 1,
)
