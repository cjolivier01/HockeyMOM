cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/external/hugin/CMakeModules")

project(hockeymom)

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 14)

# OpenCV config
SET(BUILD_opencv_apps OFF)
#SET(BUILD_opencv_freetype OFF)
#SET(BUILD_SHARED_LIBS OFF)
SET(BUILD_TESTS OFF)
SET(BUILD_PERF_TESTS OFF)
SET(BUILD_DOCS OFF)
# SET(PYTHON3_LIMITED_API ON)
SET(BUILD_OPENEXR ON)
# Mine
SET(CUDA_NVCC_FLAGS "-Xcompiler ,\"-std=c++14\"")
# SET(CMAKE_C_COMPILER gcc)
# SET(CMAKE_CXX_COMPILER g++)
# SET(INSTALL_PYTHON_EXAMPLES OFF)
# SET(INSTALL_C_EXAMPLES OFF)
# SET(EIGEN3_INCLUDE_PATH /usr/include/eigen3)
SET(OPENCV_ENABLE_NONFREE ON)
SET(WITH_CUDA ON)
SET(WITH_CUDNN ON)
SET(OPENCV_DNN_CUDA ON)
SET(ENABLE_FAST_MATH 1)
SET(CUDA_FAST_MATH 1)
SET(CUDA_ARCH_BIN 7.5)
SET(WITH_CUBLAS 1)
SET(OPENCV_DNN_CUDA 0)
# SET(WITH_EIGEN 0)
# SET(HAVE_opencv_python3 ON)
SET(BUILD_EXAMPLES OFF)
#SET(OPENCV_EXTRA_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/opencv-python/opencv_contrib/modules")
#SET(OPENCV_EXTRA_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/opencv_contrib/modules")
#MESSAGE(FATAL " OPENCV_EXTRA_MODULES_PATH=${OPENCV_EXTRA_MODULES_PATH}")
SET(OPENCV_EXTRA_MODULES_PATH "$ENV{HOME}/src/hockeymom/external/opencv-python/opencv_contrib/modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

#
#
# OPTIMIZATIONS TURNED OFF!!
#
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

find_package(PythonInterp REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED)
if (PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message("Using default python version: " ${PYTHONLIBS_VERSION_STRING})
endif()

set(HAS_UNINSTALL_TARGET 1)

set(HOCKEYMOM_TOP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HOCKEYMOM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(external/pybind11)
include_directories(SYSTEM external/pybind11/include)
include_directories(SYSTEM external/eigen)
include_directories(SYSTEM /home/colivier/.conda/envs/HockeyMOM/include)

# Very top dir is include dir start for project-wide relative imports
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# And the external dir next
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

# Include hugin sources as include dirs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/hugin/src/hugin_base)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

FIND_PACKAGE(FLANN REQUIRED)
SET(FLANN_FOUND 1)

add_subdirectory(external/eigen)
add_subdirectory(external/abseil-cpp)
#add_subdirectory(external/opencv-python/opencv)
add_subdirectory(external/hugin)

add_subdirectory(hockeymom)

