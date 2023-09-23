cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/external/hugin/CMakeModules")

project(hockeymom)

set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" REQUIRED )
if (PYTHONLIBS_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message("Using default python version: " ${PYTHONLIBS_VERSION_STRING})
endif()

set(HAS_UNINSTALL_TARGET 1)

add_subdirectory(external/pybind11)
include_directories(SYSTEM external/pybind11/include)
include_directories(SYSTEM external/eigen)
include_directories(SYSTEM /home/colivier/.conda/envs/HockeyMOM/include)

add_subdirectory(external/eigen)
#add_subdirectory(external/opencv)
#add_subdirectory(external/enblend-enfuse)
add_subdirectory(external/hugin)
add_subdirectory(hockeymom)

