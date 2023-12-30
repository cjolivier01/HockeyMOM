
# Find the yaml-cpp package
find_package(yaml-cpp REQUIRED)

#find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)
#message(FATAL_ERROR "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})

# Include certain OpenCV modules
# SET(BASE_OPENCV_DIR "${HOCKEYMOM_TOP_SRC_DIR}/external/opencv-python/opencv")
# include_directories(${BASE_OPENCV_DIR}/modules/core/include)
# include_directories(${BASE_OPENCV_DIR}/modules/videoio/include)
# include_directories(${BASE_OPENCV_DIR}/modules/gapi/include)
# include_directories(${HOCKEYMOM_BINARY_DIR})

set(
  SOURCES
  csrc/camera/Camera.cpp
  csrc/mblend/mblend.cpp
  csrc/stitcher/HmNona.cpp
  csrc/common/Gpu.cpp
  csrc/dataloader/StitchingDataLoader.cpp
  csrc/postprocess/ImagePostProcess.cpp
  csrc/pytorch/image_remap.cpp
)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/src/pytorch/share/cmake/Torch")

# set(
#   PROTO_FILES
#   csrc/mblend/mblend.proto
# )

# Generate protobuf files
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Make sure generated files are visible to compiler
include_directories(${CMAKE_CURRENT_BINARY_DIR})


add_library(hm SHARED ${SOURCES} ${PROTO_SRCS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -msse4.1 -pthread")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseWithDebugInfo")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -O0 -g")
endif()

add_link_options(-lm -lpng -ltiff -ljpeg -lstdc++)

set(
  ABSL_LIBRARIES
  absl::base
  absl::flags
  absl::stacktrace
  absl::symbolize
  absl::synchronization
  absl::str_format
  absl::time
  absl::strings
)

#
# PyTorch
#

#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#include_directories(${TORCH_INCLUDE_DIRS})


target_link_libraries(hm PUBLIC ${ABSL_LIBRARIES})
target_link_libraries(hm PRIVATE yaml-cpp)

# alphamat aruco bgsegm bioinspired calib3d ccalib core cudaarithm cudabgsegm
# cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow
# cudastereo cudawarping cudev datasets dnn dnn_objdetect dnn_superres dpm face
# features2d flann freetype fuzzy gapi hdf hfs highgui img_hash imgcodecs imgproc
# intensity_transform line_descriptor mcc ml objdetect optflow phase_unwrapping
# photo plot python3 quality rapid reg rgbd saliency shape stereo stitching
# structured_light superres surface_matching text tracking
# video videoio videostab wechat_qrcode xfeatures2d ximgproc xobjdetect xphoto

# set(OPENCV_MODULES videoio opencv_imgproc opencv_imgcodecs)
# target_link_libraries(hm PRIVATE ${OPENCV_MODULES})

set(CMAKE_INSTALL_RPATH "$ORIGIN")

target_include_directories(hm PUBLIC ./include)
set_target_properties(hm PROPERTIES INSTALL_RPATH "$ORIGIN")
target_link_libraries(hm PRIVATE ${HM_HUGIN_COMMON_LIBS} huginbase)
#target_link_libraries(hm PRIVATE OpenGL::OpenGL)
target_link_libraries(hm PRIVATE glfw)
target_link_libraries(hm PUBLIC ${TORCH_LIBRARIES})

include_directories(${HOCKEYMOM_TOP_SRC_DIR}/external/hugin/src)
include_directories(${HOCKEYMOM_BINARY_DIR}/src)

file(GLOB_RECURSE PYXMOLPP_SOURCES
        src/**.cpp
        src/**.h
)
pybind11_add_module(_hockeymom ${PYXMOLPP_SOURCES})
target_include_directories(_hockeymom PRIVATE ${CMAKE_CURRENT_LIST_DIR}/_hockeymom)
target_link_libraries(_hockeymom PUBLIC hm)
target_link_libraries(_hockeymom PRIVATE ${ABSL_LIBRARIES})
set_target_properties(_hockeymom PROPERTIES INSTALL_RPATH "$ORIGIN")

add_subdirectory(csrc)
