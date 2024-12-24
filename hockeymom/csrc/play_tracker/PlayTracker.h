#pragma once

#include "hockeymom/csrc/play_tracker/LivingBox.h"
#include "hockeymom/csrc/play_tracker/PlayDetector.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hm {
namespace play_tracker {

struct PlayTrackerConfig {
  std::vector<AllLivingBoxConfig> living_boxes;
  // After this number of ticks, "lost" tracks are discarded
  size_t max_lost_track_age{30};
  bool ignore_largest_bbox{true};
};

struct PlayTrackerState {
  std::unordered_map<size_t /*track_id*/, PlayerSTrack> player_tracks;
};

struct PlayTrackerResults {
  std::vector<BBox> cluster_boxes;
  std::vector<BBox> tracking_boxes;
  PlayDetectorResult play_detection;
};

class PlayTracker : public IBreakawayAdjuster {

  using Velocity = PointDiff;

 public:
  PlayTracker(const BBox& initial_box, const PlayTrackerConfig& config);
  virtual ~PlayTracker() = default;

  PlayTrackerResults forward(
      std::vector<size_t>& tracking_ids,
      std::vector<BBox>& tracking_boxes);
  
  void set_bboxes(const std::vector<BBox>& bboxes);
  void set_bboxes_scaled(BBox bbox, float scale_step);

 private:
  void create_boxes(const BBox& initial_box);
  BBox get_cluster_box(const std::vector<BBox>& tracking_boxes) const;

  // BEGIN IBreakawayAdjuster

  // END IBreakawayAdjuster

  //
  // DATA
  //

  // Config
  const PlayTrackerConfig config_;

  // Cluster stuff
  const std::array<size_t, 2> cluster_sizes_{2, 3};

  // Tracking stuff
  std::vector<std::unique_ptr<ILivingBox>> living_boxes_;

  // Housekeeping stuff
  // Just keep track of the number of iterations
  size_t tick_count_{0};

  // Should be close to last since it has a pointer to this class
  PlayDetector play_detector_;
};

} // namespace play_tracker
} // namespace hm
