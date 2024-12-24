
#include "hockeymom/csrc/play_tracker/PlayDetector.h"

#include <cassert>
#include <set>
#include <stdexcept>

namespace hm {
namespace play_tracker {

namespace {

using Velocity = PointDiff;

// Helper function to compute the magnitude of a velocity
double compute_magnitude(const Velocity& velocity) {
  return std::sqrt(velocity.dx * velocity.dx + velocity.dy * velocity.dy);
}

// Function to compute the average velocity of the top N fastest-moving
// velocities with the same dx sign
Velocity average_velocity_of_fastest_n_in_same_x_direction(
    const std::vector<Velocity>& velocities,
    int n) {
  if (n <= 0) {
    throw std::invalid_argument("n must be greater than 0.");
  }

  // Filter velocities with the same sign of dx (use the sign of the first
  // velocity's dx as the reference)
  std::vector<Velocity> filtered_velocities;
  bool is_positive = velocities[0].dx > 0; // Assume the sign of dx to filter by
  for (const auto& velocity : velocities) {
    if ((is_positive && velocity.dx > 0) || (!is_positive && velocity.dx < 0)) {
      filtered_velocities.push_back(velocity);
    }
  }

  if (filtered_velocities.size() < static_cast<size_t>(n)) {
    throw std::invalid_argument(
        "Not enough velocities with the same dx sign to compute the average.");
  }

  // Sort filtered velocities by magnitude in descending order
  std::sort(
      filtered_velocities.begin(),
      filtered_velocities.end(),
      [](const Velocity& a, const Velocity& b) {
        return compute_magnitude(a) > compute_magnitude(b);
      });

  // Compute the average dx and dy for the top n velocities
  double total_dx = 0.0, total_dy = 0.0;
  for (int i = 0; i < n; ++i) {
    total_dx += filtered_velocities[i].dx;
    total_dy += filtered_velocities[i].dy;
  }

  return {.dx = float(total_dx / n), .dy = float(total_dy / n)};
}

} // namespace

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
    std::vector<BBox>& tracking_boxes,
    const std::set<size_t>& disregard_tracking_ids) {
  assert(tracking_ids.size() == tracking_boxes.size());

  PointDiff cumulative_velocity{.dx = 0.0, .dy = 0.0};
  std::vector<Velocity> velocities;

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
                            PlayerSTrack(
                                config_.max_positions,
                                config_.max_velocity_positions,
                                config_.frame_step))
                        .first;
    }
    PlayerSTrack& track = found_track->second;
    track.add_position(frame_id, tracking_boxes.at(i).center());
    if (!disregard_tracking_ids.count(track_id)) {
      auto track_velocity = track.velocity();
      velocities.emplace_back(track_velocity);
      cumulative_velocity = cumulative_velocity + track_velocity;
    }
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

  //
  // Ok, now analyze what's going on...
  //
}

} // namespace play_tracker
} // namespace hm
