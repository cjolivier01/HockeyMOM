#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"
#include "hockeymom/csrc/play_tracker/PlayerSTrack.h"

#include <set>
#include <unordered_map>
#include <vector>

namespace hm {
namespace play_tracker {

struct IBreakawayAdjuster {
  virtual ~IBreakawayAdjuster() = default;
};

struct PlayDetectorConfig {
  size_t max_positions{30};
  size_t max_velocity_positions{10};
  size_t frame_step{1};
};

class PlayDetector {
 public:
  PlayDetector(const PlayDetectorConfig& config, IBreakawayAdjuster* adjuster);

  void forward(
      size_t frame_id,
      std::vector<size_t>& tracking_ids,
      std::vector<BBox>& tracking_boxes,
      const std::set<size_t>& disregard_tracking_ids);

  void reset();

 private:
  const PlayDetectorConfig config_;
  IBreakawayAdjuster* adjuster_;

  using TrackingMap = std::unordered_map</*tracking_id=*/size_t, PlayerSTrack>;

  TrackingMap tracks_;
};

} // namespace play_tracker
} // namespace hm
