#
#
#

load("@com_extension_dev//build:graph_extension.bzl", "graph_cc_extension")
load("@com_extension_dev//build:registry.bzl", "register_extension")

exports_files(["LICENSE"])

graph_cc_extension(
    name="hello_world",
    srcs=[
        "HelloWorld.cpp",
        "hello_world.cpp",
    ],
    hdrs=[
        "HelloWorld.hpp",
    ],
    deps=[
        "@StandardExtension",
    ],
)

register_extension(
    name="register_helloworld_ext",
    badges=[""],
    extension="hello_world",
    labels=[
        "nvidia",
        "gpu",
    ],
    ngc_dependencies={
        "StandardExtension": "2.6.0",
    },
    priority="1",
    git_repository="",
    url="www.example.com",
    uuid="05f887af-ddee-3d15-b467-168585f324f4",
    version="1.0.2",
    visibility=["//visibility:public"],
)
