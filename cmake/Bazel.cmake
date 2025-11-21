# Function: load_bazel_target_vars
#
# Given the WORKSPACE directory and a Bazel target label (e.g. "//foo/bar:lib"),
# this function does the following:
#   - Queries Bazel to get the output directory.
#   - Converts the Bazel label to a relative path.
#   - Extracts the actual target name (the part after the colon).
#   - Sets the following variables in the parent scope:
#         <target_name>_BINARY       - full path to the generated binary or library.
#         <target_name>_INCLUDE_DIRS - list of include directories.
#         <target_name>_LIBRARIES    - list of libraries to link.
#
# Example usage:
#   load_bazel_target_vars("${CMAKE_CURRENT_SOURCE_DIR}" "//foo/bar:lib")
#
function(load_bazel_target_vars WORKSPACE_DIR BAZEL_TARGET)
  # Get the output base directory from Bazel (e.g. bazel-bin).
  execute_process(
    COMMAND bazel info bazel-bin
    WORKING_DIRECTORY ${WORKSPACE_DIR}
    OUTPUT_VARIABLE BAZEL_BIN_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Convert the Bazel target label into a relative path.
  # Remove the leading "//" and replace ":" with "/".
  string(REGEX REPLACE "^//" "" TARGET_REL "${BAZEL_TARGET}")
  string(REPLACE ":" "/" TARGET_REL "${TARGET_REL}")

  # Determine the platform-specific shared library extension.
  if(WIN32)
    set(LIB_EXT ".dll")
  elseif(APPLE)
    set(LIB_EXT ".dylib")
  else()
    set(LIB_EXT ".so")
  endif()

  # Construct the full path to the binary/library.
  set(TARGET_BINARY "${BAZEL_BIN_DIR}/${TARGET_REL}${LIB_EXT}")

  # Extract the actual target name from the Bazel label.
  # For a label like "//foo/bar:lib", TARGET_NAME will be "lib".
  string(REGEX REPLACE ".*:(.+)" "\\1" TARGET_NAME "${BAZEL_TARGET}")

  # Set variables named after the target name.
  set(${TARGET_NAME}_BINARY "${TARGET_BINARY}" PARENT_SCOPE)
  set(${TARGET_NAME}_INCLUDE_DIRS "${WORKSPACE_DIR}" PARENT_SCOPE)
  set(${TARGET_NAME}_LIBRARIES "${TARGET_BINARY}" PARENT_SCOPE)

  message(STATUS "Bazel target '${BAZEL_TARGET}' resolved as target name '${TARGET_NAME}'")
  message(STATUS "${TARGET_NAME}_BINARY = ${TARGET_BINARY}")
endfunction()

cmake_minimum_required(VERSION 3.12)
project(MyProject)

#
# Usage Example
#
# Call the function.
# load_bazel_target_vars("${CMAKE_CURRENT_SOURCE_DIR}" "//foo/bar:lib")
#
# Now, if the target name is "lib", the following variables have been defined:
#   lib_BINARY
#   lib_INCLUDE_DIRS
#   lib_LIBRARIES
#include_directories(${lib_INCLUDE_DIRS})
#link_libraries(${lib_LIBRARIES})

#add_executable(my_exe src/main.cpp)
#target_link_libraries(my_exe ${lib_LIBRARIES})
