file(GLOB_RECURSE SOURCES csrc/**.cpp)
add_library(hm SHARED ${SOURCES})
target_include_directories(hm PUBLIC ./include)

file(GLOB_RECURSE PYXMOLPP_SOURCES
        src/**.cpp
        src/**.h
)
pybind11_add_module(_hockeymom ${PYXMOLPP_SOURCES})
target_include_directories(_hockeymom PRIVATE ${CMAKE_CURRENT_LIST_DIR}/_hockeymom)
target_link_libraries(_hockeymom PRIVATE hm)
