import argparse
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion
from typing import Tuple

import torch
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# from setuptools.command.build_ext import build_ext


_ARGS = None


def run_submodule_setup(submodule_subdir: str):
    current_dir = os.getcwd()
    os.chdir(submodule_subdir)
    try:
        subprocess_command = [sys.executable, "setup.py", sys.argv[2:]]
        subprocess.run(subprocess_command)
    finally:
        os.chdir(current_dir)


def run_setup_in_directory(directory: str) -> Tuple[int, str, str]:
    """
    Runs setup.py in the specified directory with the same arguments that were
    passed to the current script.

    Args:
        directory (str): The path to the directory containing setup.py.

    Returns:
        Tuple[str, str]: The stdout and stderr output from the setup.py execution.
    """
    # Build the path to setup.py in the target directory
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        setup_path = "./setup.py"

        command = None
        # Ensure setup.py exists in the specified directory
        if not os.path.isfile(setup_path):
            if os.path.exists("pyproject.toml") or os.path.exists("requirements.txt"):
                command = ["pip", "install", "."]
            else:
                raise FileNotFoundError(f"setup.py not found in directory: {directory}")

        # Construct the command to run setup.py with the same arguments as this script
        if not command:
            command = [sys.executable, setup_path] + sys.argv[
                1:
            ]  # sys.argv[0] is the script name itself

        # Run the setup.py with the arguments
        result = subprocess.run(command, capture_output=False, text=True)

        # Output results
        if result.returncode != 0:
            print("setup.py failed with the following error:")
        # Return the stdout and stderr output for further processing
        return result.returncode, result.stdout, result.stderr
    finally:
        os.chdir(cwd)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(BuildExtension):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
        if cmake_version < LooseVersion("3.5.0"):
            raise RuntimeError("CMake >= 3.5.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        Torch_DIR = os.path.dirname(torch.__file__)
        os.environ["Torch_DIR"] = Torch_DIR

        default_build_type = "Release"
        if int(os.environ.get("DEBUG", "0")) > 0:
            default_build_type = "Debug"
        elif int(os.environ.get("PROFILE", "0")) > 0:
            default_build_type = "ReleaseWithDebugInfo"
        elif int(os.environ.get("RELEASE", "0")) > 0:
            default_build_type = "Release"

        build_type = os.environ.get("BUILD_TYPE", default_build_type)
        build_args = ["--config", build_type]

        print(f"Build type: {build_type}")

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]

        cudnn_base_path = os.path.join(os.environ["HOME"], "cudnn")
        if not os.path.exists(cudnn_base_path):
            cudnn_base_path = "/usr/local/cudnn"
            if not os.path.exists(cudnn_base_path):
                cudnn_base_path = "/usr/local/cuda"
        if os.path.exists(cudnn_base_path):
            cmake_args += [f"-DCUDNN_INCLUDE_DIR={cudnn_base_path}/include"]
            cmake_args += [f"-DCUDNN_INCLUDE_PATH={cudnn_base_path}/include"]
            cmake_args += [f"-DCUDNN_LIBRARY={cudnn_base_path}/lib/libcudnn.so"]
            cmake_args += [f"-DCUDNN_LIBRARY_PATH={cudnn_base_path}/lib/libcudnn.so"]

        cmake_args += [f"-DCMAKE_PREFIX_PATH={os.path.join(os.getcwd(), 'cmake')}"]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(build_type.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + build_type]
            build_args += ["--", "-j30"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(" ".join(["cmake", ext.sourcedir] + cmake_args))
        print(env)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        cmake_targets = [
            "--target",
            os.path.basename(ext.name),
        ]
        print(" ".join(["cmake"] + ["--build", "."] + cmake_targets + build_args))
        subprocess.check_call(
            ["cmake"] + ["--build", "."] + cmake_targets + build_args,
            cwd=self.build_temp,
        )


if __name__ == "__main__":
    if False:
        submodule_dirs = [
            "external/fast_pytorch_kmeans",
            # "xmodels/LightGlue",
        ]
        for submodule in submodule_dirs:
            print("***************************************")
            print(f"* SETUP: {submodule} *")
            print("***************************************")
            retcode, stdout, stderr = run_setup_in_directory(submodule)
            if retcode != 0:
                print(stdout)
                print(stderr)
                sys.exit(retcode)
        print("---------------------------------------")
        print("---------------------------------------")

    setup(
        name="hockeymom",
        version="0.5.0",
        author="Christopher Olivier",
        author_email="cjolivier01@gmail.com",
        description="HockeyMOM project",
        long_description=open("README.rst").read(),
        ext_modules=[CMakeExtension("hockeymom/_hockeymom")],
        packages=find_packages(),
        cmdclass=dict(build_ext=CMakeBuild),
        url="https://github.com/cjolivier01/hockeymom2",
        zip_safe=False,
        install_requires=[],
    )
