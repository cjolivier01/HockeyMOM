
set(OPENGL_egl_LIBRARY "$ENV{CONDA_PREFIX}/aarch64-conda-linux-gnu/sysroot/usr/lib64/libEGL_mesa.so.0")
set(OPENGL_gl_LIBRARY "$ENV{CONDA_PREFIX}/lib/libglapi.so")
set(OPENGL_EGL_INCLUDE_DIR "$ENV{CONDA_PREFIX}/aarch64-conda-linux-gnu/sysroot/usr/include")
set(OPENGL_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")

mark_as_advanced(
  OPENGL_INCLUDE_DIR
  OPENGL_xmesa_INCLUDE_DIR
  OPENGL_egl_LIBRARY
  OPENGL_glu_LIBRARY
  OPENGL_glx_LIBRARY
  OPENGL_gl_LIBRARY
  OPENGL_opengl_LIBRARY
  OPENGL_EGL_INCLUDE_DIR
  OPENGL_GLX_INCLUDE_DIR
)

#add_library(OpenGL::OpenGL ALIAS ${OPENGL_gl_LIBRARY})
