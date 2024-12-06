#pragma once

#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <vector>

namespace hm {
namespace play_tracker {

class PlayTracker {
 public:
  PlayTracker();
  virtual ~PlayTracker() = default;

  void create_boxes();

 private:
  std::vector<std::unique_ptr<ILivingBox>> living_boxes_;
};

} // namespace play_tracker
} // namespace hm
