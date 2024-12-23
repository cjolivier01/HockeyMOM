#include "hockeymom/csrc/play_tracker/PlayTracker.h"
#include "hockeymom/csrc/play_tracker/LivingBoxImpl.h"
#include "hockeymom/csrc//kmeans/kmeans.h"

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

BBox PlayTracker::get_cluster_box(
    const std::vector<BBox>& tracking_boxes) const {
  BBox result_box;
  size_t counter = 0;
  std::vector<float> points;
  points.reserve(tracking_boxes.size() << 1);

  for (const auto& box : tracking_boxes) {
    const Point c = box.center();
    points.push_back(c.x);
    points.push_back(c.y);
  }

  for (size_t cluster_id = 0, cluster_count = cluster_sizes_.size(); cluster_id < cluster_count; ++cluster_id) {

  }
  // std::unordered_map<size_t,
  return result_box;
}

PlayTrackerResults PlayTracker::forward(
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes) {
  assert(tracking_ids.size() == living_boxes_.size());

  PlayTrackerResults results;
  for (std::size_t i = 0; i < living_boxes_.size(); ++i) {
    auto& living_box = living_boxes_[i];

    // auto& player_track = living_box->player_track();
    // if (player_track.age(tick_count_) > config_.max_lost_track_age) {
    //   player_track.reset();
    // }
    // if (tracking_ids[i] != 0) {
    //   player_track.add_position(tick_count_, tracking_boxes[i].center());
    // }
  }
  ++tick_count_;
  return results;
}

} // namespace play_tracker
} // namespace hm
