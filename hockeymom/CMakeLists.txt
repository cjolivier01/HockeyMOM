#file(GLOB_RECURSE SOURCES csrc/**.cpp)
set(CMAKE_BUILD_TYPE Debug)
set(
  SOURCES
  csrc/mblend/mblend.cpp
  csrc/sublibA/add.cpp
  csrc/ImagePostProcess.cpp
)
#include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
add_library(hm SHARED ${SOURCES})
#add_compile_options(-std=c++14 -msse4.1 -pthread -ffast-math -Ofast)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -pthread -ffast-math -Ofast")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

add_link_options(-lm -lpng -ltiff -ljpeg -lstdc++)
target_include_directories(hm PUBLIC ./include)

file(GLOB_RECURSE PYXMOLPP_SOURCES
        src/**.cpp
        src/**.h
)
pybind11_add_module(_hockeymom ${PYXMOLPP_SOURCES})
target_include_directories(_hockeymom PRIVATE ${CMAKE_CURRENT_LIST_DIR}/_hockeymom)
target_link_libraries(_hockeymom PRIVATE hm)
