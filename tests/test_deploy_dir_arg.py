import argparse


def should_parse_deploy_dir_argument():
    from hmlib.hm_opts import hm_opts

    parser = hm_opts.parser(argparse.ArgumentParser())
    args = parser.parse_args(["--deploy-dir", "/tmp/hm-deploy"])
    assert args.deploy_dir == "/tmp/hm-deploy"

