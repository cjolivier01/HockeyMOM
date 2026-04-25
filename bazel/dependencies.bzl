load("@bazel_tools//tools/build_defs/repo:utils.bzl", "workspace_and_buildfile")

def _detect_py_version_from_conda_layout(ctx, conda_root):
    lib_dir = ctx.path(conda_root + "/lib")
    if not lib_dir.exists:
        return None

    best_key = None
    best_version = None
    for entry in lib_dir.readdir():
        name = entry.basename
        if not name.startswith("python"):
            continue
        version = name[len("python"):]
        parts = version.split(".")
        if len(parts) < 2:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if not ctx.path(str(entry) + "/site-packages").exists:
            continue
        key = (int(parts[0]), int(parts[1]))
        if best_key == None or key > best_key:
            best_key = key
            best_version = "%s.%s" % (parts[0], parts[1])
    return best_version

def _resolve_python_bin(ctx, conda_root):
    candidates = []

    env_python = ctx.os.environ.get("PYTHON_BIN_PATH")
    if env_python:
        candidates.append(env_python)

    candidates.append(conda_root + "/bin/python")
    candidates.append(conda_root + "/bin/python3")

    python_on_path = ctx.which("python")
    if python_on_path != None:
        candidates.append(str(python_on_path))

    python3_on_path = ctx.which("python3")
    if python3_on_path != None:
        candidates.append(str(python3_on_path))

    seen = {}
    for candidate in candidates:
        if not candidate or seen.get(candidate):
            continue
        seen[candidate] = True
        if ctx.path(candidate).exists:
            return candidate

    return None

def _detect_py_version(ctx, conda_root):
    py_ver = _detect_py_version_from_conda_layout(ctx, conda_root)
    if py_ver:
        return py_ver

    python_bin = _resolve_python_bin(ctx, conda_root)
    if python_bin == None:
        fail("Could not detect Python version: no interpreter found. Checked PYTHON_BIN_PATH, CONDA_PREFIX/bin/python, CONDA_PREFIX/bin/python3, python, and python3.")

    # run Python to get "3.12", "3.11", etc
    result = ctx.execute([
        python_bin,
        "-c",
        "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")",
    ])
    if result.return_code != 0:
        fail("Could not detect Python version from " + python_bin + ": " + result.stderr)
    return result.stdout.strip()

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
    if not ctx.path(conda_root).exists:
        fail("CONDA_PREFIX does not exist: " + conda_root)

    # 1) figure out what Python X.Y we’re on,
    #    so users don’t have to hard-code “3.12” in WORKSPACE
    py_ver = _detect_py_version(ctx, conda_root)

    # Retrieve rule attributes.
    pkg_dir = ctx.attr.package_dir.replace("{python}", py_ver)
    # substitute any `{python}` tokens
    file_list    = [f.replace("{python}", py_ver) for f in ctx.attr.files]
    for i, f in enumerate(ctx.attr.files):
        ctx.attr.files[i] = f.replace("{python}", py_ver)
    prefix_to_strip = ctx.attr.strip_prefix.replace("{python}", py_ver) if ctx.attr.strip_prefix else None

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
    environ = ["CONDA_PREFIX", "PYTHON_BIN_PATH"],
)
