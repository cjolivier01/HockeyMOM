"""Repository rule exposing the build host's glibc platform baseline."""


def _host_glibc_repository_impl(repository_ctx):
    result = repository_ctx.execute(["getconf", "GNU_LIBC_VERSION"])
    if result.return_code != 0:
        fail("Unable to detect glibc with getconf: {}".format(result.stderr.strip()))
    fields = [field for field in result.stdout.strip().split(" ") if field]
    if len(fields) != 2 or fields[0] != "glibc":
        fail("Unexpected getconf GNU_LIBC_VERSION output: {}".format(result.stdout.strip()))
    version = fields[1]
    components = version.split(".")
    if len(components) != 2:
        fail("Unsupported glibc version: {}".format(version))
    manylinux = "manylinux_{}_{}".format(components[0], components[1])
    repository_ctx.file("BUILD.bazel", "exports_files([\"defs.bzl\"])\n")
    repository_ctx.file(
        "defs.bzl",
        "HOST_GLIBC_VERSION = \"{}\"\nHOST_MANYLINUX_TAG = \"{}\"\n".format(
            version,
            manylinux,
        ),
    )


host_glibc_repository = repository_rule(
    implementation = _host_glibc_repository_impl,
    configure = True,
    local = True,
)
