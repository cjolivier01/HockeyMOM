#include "hockeymom/csrc/common/Gpu.h"

namespace hm {

namespace {
thread_local bool attempted = false;
thread_local bool attempted_result = false;
} // namespace

bool check_cuda_opengl() {
  if (attempted) {
    return attempted_result;
  }
  attempted = true;

  int argc = 2;
  char* argv[2] = {const_cast<char*>("python"), const_cast<char*>("-g")};
  attempted_result = hugin_utils::initGPU(&argc, &argv[0]);
  return attempted_result;

  // // Request an OpenGL 3.3 core profile context
  // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // //assert(false);
  // // Initialize GLFW and create an OpenGL context
  // if (!glfwInit()) {
  //   fprintf(stderr, "Failed to initialize GLFW\n");
  //   return false;
  // }

  // // Create an OpenGL context
  // GLFWwindow* window = glfwCreateWindow(100, 100, "OpenGL Window", NULL,
  // NULL); if (!window) {
  //   fprintf(stderr, "Failed to create GLFW window\n");
  //   glfwTerminate();
  //   return false;
  // }
  // glfwHideWindow(window);

  // // // Make the OpenGL context current
  // glfwMakeContextCurrent(window);

  // // Initialize GLEW (or equivalent)
  // GLenum err = glewInit();
  // if (err != GLEW_OK) {
  //   fprintf(stderr, "GLEW initialization error: %s\n",
  //   glewGetErrorString(err)); glfwTerminate(); return false;
  // }

  // // Query and print the OpenGL vendor string
  // const char* vendor = (const char*)glGetString(GL_VENDOR);
  // if (vendor) {
  //   printf("OpenGL Vendor: %s\n", vendor);
  //   attempted_result = true;
  //   return true;
  // } else {
  //   GLenum err = glGetError();
  //   fprintf(stderr, "Unable to retrieve OpenGL vendor string: %d\n", err);
  //   return false;
  // }
}

} // namespace hm
