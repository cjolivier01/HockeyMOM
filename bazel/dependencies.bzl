load("@bazel_tools//tools/build_defs/repo:utils.bzl", "workspace_and_buildfile")

# Returns True if the given string ends with a slash.
def _ends_with_slash(s):
    if s == "":
        return False
    return s[len(s) - 1:] == "/"

# Implementation of the conda repository rule.
def conda_repo_setup(ctx):
    # Get the conda installation root.
    conda_root = ctx.os.environ.get("CONDA_PREFIX")
    if not conda_root:
        fail("Environment variable CONDA_PREFIX is not set.")

    # Retrieve rule attributes.
    pkg_dir = ctx.attr.package_dir
    file_list = ctx.attr.files
    prefix_to_strip = ctx.attr.strip_prefix

    # If a package directory is provided, symlink its content recursively.
    if pkg_dir:
        if _ends_with_slash(pkg_dir):
            fail("Please remove the trailing slash from 'package_dir': " + pkg_dir)

        # Build the absolute path to the package.
        pkg_absolute = conda_root + "/" + pkg_dir
        # Default: strip off the conda root and the following slash.
        num_chars_to_strip = len(conda_root + "/")

        if prefix_to_strip:
            if _ends_with_slash(prefix_to_strip):
                fail("Please remove the trailing slash from 'strip_prefix': " + prefix_to_strip)
            alt_pkg_path = conda_root + "/" + prefix_to_strip
            if not ctx.path(alt_pkg_path).exists:
                fail("Cannot locate the path derived from 'strip_prefix': " + alt_pkg_path)
            if alt_pkg_path.find(pkg_absolute) != 0:
                fail("Expected 'strip_prefix' (" + alt_pkg_path +
                     ") to be under the package directory (" + pkg_absolute + ")")
            pkg_absolute = alt_pkg_path
            num_chars_to_strip = len(alt_pkg_path + "/")

        # Iterate over all entries in the package directory and symlink them.
        for entry in ctx.path(pkg_absolute).readdir():
            # Compute the destination path by removing the leading portion.
            dest = str(entry)[num_chars_to_strip:]
            ctx.symlink(entry, dest)

    # Symlink any individual files specified.
    for f in file_list:
        abs_file = conda_root + "/" + f
        if not ctx.path(abs_file).exists:
            fail("File not found: " + abs_file)
        ctx.symlink(abs_file, f)

    # Finally, generate the BUILD and WORKSPACE files in this repository.
    workspace_and_buildfile(ctx)

# The repository rule "conda_repository" creates a repository by symlinking
# content from your conda environment (under CONDA_PREFIX). It can either symlink
# an entire directory (via "package_dir") with an optional "strip_prefix" or
# individual files (via "files").
conda_repository = repository_rule(
    doc = """
        Creates a repository from a Conda environment by symlinking files and/or
        a directory located relative to CONDA_PREFIX.

        The 'package_dir' attribute (if provided) causes the entire directory's
        contents to be symlinked into the repository. An optional 'strip_prefix'
        can adjust the subdirectory layout.

        Additionally, individual files listed in 'files' are symlinked directly.

        Example usage in WORKSPACE:

        load("//:conda_repo.bzl", "conda_repository")
        conda_repository(
            name = "conda_lzma",
            package_dir = "include/lzma",
            files = [
                "include/lzma.h",
                "lib/liblzma.so"
            ],
            build_file = "@//third_party/bazel-builds:conda_lzma.BUILD",
        )
    """,
    implementation = conda_repo_setup,
    attrs = {
        "package_dir": attr.string(),
        "files": attr.string_list(),
        "build_file": attr.label(allow_single_file = True),
        "build_file_content": attr.string(),
        "workspace_file": attr.label(allow_single_file = True),
        "workspace_file_content": attr.string(),
        "strip_prefix": attr.string(),
    },
    environ = ["CONDA_PREFIX"],
)
