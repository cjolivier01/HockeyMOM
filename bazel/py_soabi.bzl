def _py_soabi_impl(repository_ctx):
    # Execute Python to get the SOABI value.
    result = repository_ctx.execute(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_config_var('SOABI'))"],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to run python command: " + result.stderr)
    soabi = result.stdout.strip()

    result = repository_ctx.execute(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_platform())"],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to run python command: " + result.stderr)
    platform = result.stdout.strip()

    result = repository_ctx.execute(
        ["python3", "-c", 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")'],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to run python command: " + result.stderr)
    abi = result.stdout.strip()

    # print(f"cp{sys.version_info.major}{sys.version_info.minor}")

    # Write the soabi.bzl file containing the SOABI variable.
    file_lines = "SOABI = \"%s\"\n" % soabi
    file_lines += "PLATFORM = \"%s\"\n" % platform
    file_lines += "ABI = \"%s\"\n" % abi
    # repository_ctx.file("soabi.bzl", "SOABI = \"%s\"\n" % soabi)
    print(file_lines)
    repository_ctx.file("soabi.bzl", file_lines)
    
    # Also create an empty BUILD file so that the repository is a valid package.
    repository_ctx.file("BUILD", "")

py_soabi = repository_rule(
    implementation = _py_soabi_impl,
    local = True,
)
