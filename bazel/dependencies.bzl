load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "workspace_and_buildfile",
)
PYTORCH_LOCAL_DIR = "../"

def _has_trailing_slash(str):
    if not str:
        return False
    str_len = len(str)
    last_char_of_str = str[str_len-1:]
    return (last_char_of_str == "/")


def conda_repository_impl(repo_ctx):
    conda_prefix = repo_ctx.os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        fail("CONDA_PREFIX not defined")
    package_dir = repo_ctx.attr.package_dir
    files = repo_ctx.attr.files
    strip_prefix = repo_ctx.attr.strip_prefix


    root = repo_ctx.path(".")

    # Recursively symlink files specified in package_dirs.
    if package_dir:
        if _has_trailing_slash(package_dir):
            fail("Remove trailing slash from package_dir {}".format(package_dir))

        package_link = "{}/{}".format(conda_prefix, package_dir)
        num_leading_characters_to_strip = len(conda_prefix + "/")

        if strip_prefix:
            if _has_trailing_slash(strip_prefix):
                fail("Remove trailing slash from strip_prefix {}".format(strip_prefix))
            stripped_link = "{}/{}".format(conda_prefix, strip_prefix)
            if not repo_ctx.path(stripped_link).exists:
                fail("Could not find path {} in {}".format(stripped_link, package_link))
            if stripped_link.find(package_link) != 0:
                fail("Expected strip_prefix {} to be at or under package_link {}".format(stripped_link, package_link))
            package_link = stripped_link
            num_leading_characters_to_strip = len(stripped_link + "/")

        for item in repo_ctx.path(package_link).readdir():
            dest_path = str(item)[num_leading_characters_to_strip:]
            repo_ctx.symlink(item, dest_path)
    
    # Copy individual files.
    for file in files:
        file_link = "{}/{}".format(conda_prefix, file)
        if not repo_ctx.path(file_link).exists:
            fail("could not find file {}".format(file_link))
        repo_ctx.symlink(file_link, file)

    workspace_and_buildfile(repo_ctx)
  

conda_repository = repository_rule(
    doc = """
    Create a conda repository by copying files and/or a directory from a path
    relative to the environment variable CONDA_PREFIX.

    The files in package_dir are symlinked recursively to the build context.
    The strip_prefix field applies to the package_dir.

    Individual files are symlinked to the build context too. Individual files
    can be used instead of a directory when the files needed to build
    things is spread throughout a conda env.
    Examples include individual headers in include, or shared libraries.

    Example usage:
    In the WORKSPACE file:
    load("//bazel:conda_library.bzl", "conda_repository")
    conda_repository(
        name = "conda_lzma",
        package_dir = "include/lzma",
        files = [
            "include/lzma.h",
            "lib/liblzma.so"
        ],
        build_file = "@//third_party/bazel-builds:conda_lzma.BUILD",
    )

    In the third_party/bazel-builds/conda_lzma.BUILD file:
    filegroup(
        name = "lzma_private_headers",
        srcs = glob(["include/lzma/*"])
    )
    cc_library(
        name = "lzma",
        hdrs = [
            ":lzma_private_headers",
        "include/lzma.h"
        ],
        srcs = ["lib/liblzma.so"],
        visibility = ["//visibility:public"]
    )
    """,
    implementation = conda_repository_impl,
    attrs = dict(
        package_dir = attr.string(),
        files = attr.string_list(),
        build_file = attr.label(allow_single_file = True),
        build_file_content = attr.string(),
        workspace_file = attr.label(allow_single_file = True),
        workspace_file_content = attr.string(),
        strip_prefix = attr.string()
    ),
    environ = ["CONDA_PREFIX"]
)
