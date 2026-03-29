cc_library(
    name = "glibconfig",
    hdrs = glob([
        "lib/*/glib-2.0/include/glibconfig.h",
        "lib/glib-2.0/include/glibconfig.h",
        "lib64/glib-2.0/include/glibconfig.h",
    ]),
    includes = [
        "lib/glib-2.0/include",
        "lib64/glib-2.0/include",
        "lib/x86_64-linux-gnu/glib-2.0/include",
        "lib/aarch64-linux-gnu/glib-2.0/include",
    ],
    visibility = ["//visibility:public"],
)
