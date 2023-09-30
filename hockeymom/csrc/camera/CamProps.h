#pragma once

#include <cstdint>

namespace hm {
namespace camera {
struct Box {
  int left{0};
  int top{0};
  int right{0};
  int bottom{0};
};

struct CameraProperties {};

struct CameraBehavior {};

struct CameraState {};

} // namespace camera
} // namespace hm
