def _py_soabi_impl(repository_ctx):
    # Execute Python to get the SOABI value.
    result = repository_ctx.execute(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_config_var('SOABI'))"],
        quiet = True,
    )
    if result.return_code != 0:
        fail("Failed to run python command: " + result.stderr)
    soabi = result.stdout.strip()
    
    # Write the soabi.bzl file containing the SOABI variable.
    repository_ctx.file("soabi.bzl", "SOABI = \"%s\"\n" % soabi)
    
    # Also create an empty BUILD file so that the repository is a valid package.
    repository_ctx.file("BUILD", "")

py_soabi = repository_rule(
    implementation = _py_soabi_impl,
    local = True,
)
