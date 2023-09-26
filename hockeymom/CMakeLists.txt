
# Include certain OpenCV modules
SET(BASE_OPENCV_DIR "${HOCKEYMOM_TOP_SRC_DIR}/external/opencv-python/opencv")
include_directories(${BASE_OPENCV_DIR}/modules/core/include)
include_directories(${BASE_OPENCV_DIR}/modules/videoio/include)
include_directories(${BASE_OPENCV_DIR}/modules/gapi/include)
include_directories(${HOCKEYMOM_BINARY_DIR})

set(
  SOURCES
  csrc/mblend/mblend.cpp
  csrc/stitcher/HmNona.cpp
  csrc/dataloader/StitchedDataLoader.cpp
  csrc/ImagePostProcess.cpp
)
add_library(hm SHARED ${SOURCES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -msse4.1 -pthread -ffast-math -Ofast")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

add_link_options(-lm -lpng -ltiff -ljpeg -lstdc++)

target_include_directories(hm PUBLIC ./include)
target_link_libraries(hm PRIVATE ${HM_HUGIN_COMMON_LIBS} huginbase)
target_link_libraries(hm PRIVATE OpenGL::OpenGL)
target_link_libraries(hm PRIVATE glfw)

include_directories(${HOCKEYMOM_TOP_SRC_DIR}/external/hugin/src)
include_directories(${HOCKEYMOM_BINARY_DIR}/src)

file(GLOB_RECURSE PYXMOLPP_SOURCES
        src/**.cpp
        src/**.h
)
pybind11_add_module(_hockeymom ${PYXMOLPP_SOURCES})
target_include_directories(_hockeymom PRIVATE ${CMAKE_CURRENT_LIST_DIR}/_hockeymom)
target_link_libraries(_hockeymom PRIVATE hm)
