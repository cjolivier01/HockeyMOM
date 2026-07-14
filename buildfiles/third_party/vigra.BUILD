load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

_VIGRA_HDRS = glob([
    "include/vigra/**/*.h",
    "include/vigra/**/*.hxx",
])

cc_library(
    name = "vigraimpex",
    srcs = glob([
        "src/impex/*.c",
        "src/impex/*.cxx",
        "src/impex/*.h",
        "src/impex/*.hxx",
    ]),
    hdrs = _VIGRA_HDRS,
    copts = [
        "-DVIGRA_STATIC_LIB",
        "-DHasJPEG",
        "-DHasPNG",
        "-DHasTIFF",
        "-DHasZLIB",
    ],
    includes = [
        "include",
        "src/impex",
    ],
    linkopts = [
        "-ljpeg",
        "-lpng",
        "-ltiff",
        "-lz",
    ],
)

cc_library(
    name = "vigra",
    hdrs = _VIGRA_HDRS,
    includes = ["include"],
    deps = [":vigraimpex"],
)
