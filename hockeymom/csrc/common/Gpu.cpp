#include "hockeymom/csrc/common/Gpu.h"

#include <X11/Xlib.h> // Include Xlib.h for XInitThreads

#include <mutex>

namespace hm {

namespace {
std::mutex init_mu_;
std::size_t global_initialized{false};
thread_local bool attempted = false;
thread_local bool attempted_result = false;
} // namespace

bool check_cuda_opengl() {
  if (attempted) {
    return attempted_result;
  }

  std::unique_lock<std::mutex> lk(init_mu_);
  if (!global_initialized) {
    assert(XInitThreads());
    global_initialized = true;
  }

  attempted = true;

  int argc = 2;
  char* argv[2] = {const_cast<char*>("python"), const_cast<char*>("-g")};
  attempted_result = hugin_utils::initGPU(&argc, &argv[0]);
  return attempted_result;
}

std::size_t get_tick_count_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

void set_thread_name(const std::string& thread_name, int index) {
  if (index >= 0) {
    std::string n = thread_name;
    n += '-';
    n += std::to_string(index);
    pthread_setname_np(pthread_self(), n.c_str());
  } else {
    pthread_setname_np(pthread_self(), thread_name.c_str());
  }
}

} // namespace hm
