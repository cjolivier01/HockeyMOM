#pragma once

#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <memory>
#include <vector>

namespace hm {
namespace play_tracker {

struct PlayTrackerConfig {
  std::vector<AllLivingBoxConfig>   living_boxes;
};

struct PlayTrackerResults {

};

class PlayTracker {
 public:
  PlayTracker(const PlayTrackerConfig& config);
  virtual ~PlayTracker() = default;



 private:
  void create_boxes();

  const PlayTrackerConfig config_;
  std::vector<std::unique_ptr<ILivingBox>> living_boxes_;
};

} // namespace play_tracker
} // namespace hm
