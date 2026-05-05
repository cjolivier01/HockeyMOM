config_setting(
    name = "aarch64-linux-gnu",
    define_values = {"multiarch": "aarch64-linux-gnu"},
)

config_setting(
    name = "x86_64-linux-gnu",
    define_values = {"multiarch": "x86_64-linux-gnu"},
)

cc_library(
    name = "libsoup",
    hdrs = glob([
        "include/libsoup-2.4/**/*.h*",
        "include/nlohmann/**/*.h*",
    ]),
    includes = [
        "include/libsoup-2.4",
        "include/nlohmann",
    ],
    linkopts = select({
        ":aarch64-linux-gnu": ["-L/usr/lib/aarch64-linux-gnu"],
        ":x86_64-linux-gnu": ["-L/usr/lib/x86_64-linux-gnu"],
        "//conditions:default": [],
    }) + [
        "-l:libsoup-2.4.so",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@glib",
        "@json_glib",
    ],
)
