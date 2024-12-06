#pragma once

#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <vector>

namespace hm {
namespace play_tracker {

struct PlayTrackerConfig {
  std::vector<AllLivingBoxConfig>   living_boxes;
};

class PlayTracker {
 public:
  PlayTracker(const PlayTrackerConfig& config);
  virtual ~PlayTracker() = default;

  void create_boxes();

 private:
  const PlayTrackerConfig config_;
  std::vector<std::unique_ptr<ILivingBox>> living_boxes_;
};

} // namespace play_tracker
} // namespace hm
