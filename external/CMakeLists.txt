# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_subdirectory(eigen)
add_subdirectory(abseil-cpp)
# add_subdirectory(opencv-python/opencv)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|x86|i386")
  add_subdirectory(hugin)
endif()
