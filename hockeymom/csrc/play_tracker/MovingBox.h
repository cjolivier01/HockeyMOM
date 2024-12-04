#pragma once

#include <memory>
#include <string>

namespace hm {
namespace play_tracker {

class MovingBox;
struct MovingBoxConfig {
  std::string name;
  std::shared_ptr<MovingBox> folowing_box;
};

class MovingBox {
 public:
  MovingBox(const MovingBoxConfig& config);

 private:
  MovingBoxConfig config_;
};

} // namespace play_tracker
} // namespace hm
