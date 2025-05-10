#pragma once

#include <cuda_runtime.h>

#include "jetson-utils/display/glDisplay.h"

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace hm {
namespace display {

struct DisplaySurface {
  DisplaySurface(void* d, int w, int h, int p, int c)
      : d_ptr(d), width(w), height(h), pitch(p), channels(c) {}

  void* d_ptr;
  int width;
  int height;
  int pitch;
  int channels;

  int pitch_width() const {
    assert(pitch % (channels * width) == 0);
    return pitch / channels;
  }
};

class HmRenderSet {
 public:
  bool render(
      const std::string& name,
      const DisplaySurface& surface,
      cudaStream_t stream);

 private:
  static std::unique_ptr<glDisplay> create_video_output(
      const std::string& name,
      const DisplaySurface& surface);
  videoOutput* get_video_output(
      const std::string& name,
      const DisplaySurface& surface);

  std::mutex mu_;
  std::map<std::string, std::unique_ptr<glDisplay>> video_outputs_;
};

} // namespace display
} // namespace hm
