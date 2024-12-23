#include "hockeymom/csrc/play_tracker/PlayTracker.h"
#include "hockeymom/csrc/play_tracker/LivingBoxImpl.h"

namespace hm {
namespace play_tracker {
PlayTracker::PlayTracker(const PlayTrackerConfig& config) : config_(config) {
  create_boxes();
}

void PlayTracker::create_boxes() {
  for (std::size_t i = 0; i < config_.living_boxes.size(); ++i) {
    living_boxes_.emplace_back(std::make_unique<LivingBox>(
        std::to_string(i + 1), BBox(), config_.living_boxes[i]));
  }
}

} // namespace play_tracker
} // namespace hm
