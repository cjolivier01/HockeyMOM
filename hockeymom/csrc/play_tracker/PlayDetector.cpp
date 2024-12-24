
#include "hockeymom/csrc/play_tracker/PlayDetector.h"

#include <cassert>
#include <set>

namespace hm {
namespace play_tracker {

PlayDetector::PlayDetector(
    const PlayDetectorConfig& config,
    IBreakawayAdjuster* adjuster)
    : config_(config), adjuster_(adjuster) {}

void PlayDetector::reset() {
  tracks_.clear();
}

void PlayDetector::forward(
    size_t frame_id,
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes) {
  assert(tracking_ids.size() == tracking_boxes.size());

  PointDiff cumulative_velocity{.dx = 0.0, .dy = 0.0};
  std::multiset<float> dx_velocities, dy_velocities;

  //
  // Update the tracks
  //
  for (size_t i = 0, n = tracking_ids.size(); i < n; ++i) {
    const size_t track_id = tracking_ids[i];
    const BBox& bbox = tracking_boxes.at(i);
    auto found_track = tracks_.find(track_id);
    if (found_track == tracks_.end()) {
      found_track = tracks_
                        .emplace(
                            track_id,
                            PlayerTrack(
                                config_.max_positions,
                                config_.max_velocity_positions,
                                config_.frame_step))
                        .first;
    }
    PlayerTrack& track = found_track->second;
    track.add_position(frame_id, tracking_boxes.at(i).center());
    auto track_velocity = track.velocity();
    dx_velocities.insert(track_velocity.dx);
    dy_velocities.insert(track_velocity.dy);
    cumulative_velocity = cumulative_velocity + track_velocity;
  }
  //
  // Dump any lost, stale tracks
  //
  std::vector<TrackingMap::iterator> remove_tracks;
  remove_tracks.reserve(tracks_.size());
  for (auto track_iter = tracks_.begin(), e_iter = tracks_.end();
       track_iter != e_iter;
       ++track_iter) {
    if (track_iter->second.age(frame_id) >= config_.max_positions) {
      // Lost track is now stale
      remove_tracks.emplace_back(track_iter);
    }
  }
  std::for_each(remove_tracks.begin(), remove_tracks.end(), [this](auto& iter) {
    tracks_.erase(iter);
  });
}

} // namespace play_tracker
} // namespace hm
