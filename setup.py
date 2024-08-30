import argparse
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

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

        cmake_version = LooseVersion(
            re.search(r"version\s*([\d.]+)", out.decode()).group(1)
        )
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

        # Torch_DIR = "/home/colivier/src/pytorch/torch/share/cmake/Torch/"
        Torch_DIR = os.path.dirname(torch.__file__)
        os.environ["Torch_DIR"] = Torch_DIR
        # os.environ["CUDNN_LIBRARY_PATH"] = "/usr/local/cudnn/lib"
        # os.environ["CUDNN_LIB_DIR"] = "/usr/local/cudnn/lib"
        # os.environ["CUDNN_INCLUDE_PATH"] = "/usr/local/cudnn/include"

        default_build_type = "Release"
        # default_build_type = "ReleaseWithDebugInfo"
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

        #        cmake_args += ["-DCMAKE_C_COMPILER=clang"]
        #        cmake_args += ["-DCMAKE_CXX_COMPILER=clang++"]

        # cmake_args += ["-DHM_BUILD_ASAN=1"]

        # cmake_args += ["-DCMAKE_PREFIX_PATH="]
        # cmake_args += ["-DCMAKE_FIND_DEBUG_MODE=ON"]
        # cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        # if ARGS.conda:
        cmake_args += [f"-DCMAKE_PREFIX_PATH={os.path.join(os.getcwd(), 'cmake')}"]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    build_type.upper(), extdir
                )
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
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        cmake_targets = [
            "--target",
            os.path.basename(ext.name),
            # "--target",
            # "nona",
            # "--target",
            # "pto_gen",
            # "--target",
            # "cpfind",
            # "--target",
            # "autooptimiser",
        ]
        print(" ".join(["cmake"] + ["--build", "."] + cmake_targets + build_args))
        subprocess.check_call(
            ["cmake"] + ["--build", "."] + cmake_targets + build_args,
            cwd=self.build_temp,
        )


if __name__ == "__main__":
    # _ = _ARGS
    # parser = argparse.ArgumentParser(description='HockeyMOM Setup Script.')
    # parser.add_argument('--conda', action="store_true", help='Build using a conda environment')
    # _ARGS = parser.parse_args()

    # run_submodule_setup('DCNv2')
    # run_submodule_setup('external/fast_pytorch_kmeans')

    setup(
        name="hockeymom",
        version="0.5.0",
        author="Christopher Olivier",
        author_email="cjolivier01@gmail.com",
        description="HockeyMOM project",
        long_description=open("README.rst").read(),
        ext_modules=[
            # CppExtension("hockeymom/_hockeymom")
            CMakeExtension("hockeymom/_hockeymom")
        ],
        packages=find_packages(),
        cmdclass=dict(build_ext=CMakeBuild),
        url="https://github.com/cjolivier01/hockeymom2",
        zip_safe=False,
        install_requires=[],
    )
