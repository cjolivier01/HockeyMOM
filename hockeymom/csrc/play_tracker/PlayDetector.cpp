
#include "hockeymom/csrc/play_tracker/PlayDetector.h"

#include <cassert>
#include <iostream>
#include <iterator>
#include <set>

namespace hm {
namespace play_tracker {

namespace {

using Velocity = PointDiff;

constexpr float kLargestPosFloatValue = std::numeric_limits<FloatValue>::max();
constexpr float kLargestNegFloatValue = std::numeric_limits<FloatValue>::min();

constexpr size_t kMinPlayersForBreakawayDetection = 4;

// Helper function to compute the magnitude of a velocity
double compute_magnitude(const Velocity& velocity) {
  return std::sqrt(velocity.dx * velocity.dx + velocity.dy * velocity.dy);
}

template <typename MapType>
class ValueIterator {
  using IteratorType = typename MapType::iterator;

 public:
  using iterator_category = std::input_iterator_tag; // Iterator category
  using value_type = typename MapType::mapped_type; // Type of the values
  using difference_type = std::ptrdiff_t; // Difference type
  using pointer = value_type*; // Pointer type
  using reference = value_type&; // Reference type

  explicit ValueIterator(IteratorType it) : mapIt(it) {}

  // Dereference operators
  reference operator*() {
    return mapIt->second;
  }
  pointer operator->() {
    return &(mapIt->second);
  }

  // Pre-increment operator
  ValueIterator& operator++() {
    ++mapIt;
    return *this;
  }

  // Post-increment operator
  ValueIterator operator++(int) {
    ValueIterator temp = *this;
    ++mapIt;
    return temp;
  }

  // Equality operators
  bool operator==(const ValueIterator& other) const {
    return mapIt == other.mapIt;
  }
  bool operator!=(const ValueIterator& other) const {
    return mapIt != other.mapIt;
  }

 private:
  IteratorType mapIt;
};

// Helper function to create value iterators
template <typename MapType>
auto make_value_iterator(MapType& map) {
  return std::pair<ValueIterator<MapType>, ValueIterator<MapType>>(
      ValueIterator<MapType>(map.begin()), ValueIterator<MapType>(map.end()));
}

} // namespace

PlayDetector::PlayDetector(
    const PlayDetectorConfig& config,
    IBreakawayAdjuster* adjuster)
    : config_(config), adjuster_(adjuster) {}

void PlayDetector::reset() {
  tracks_.clear();
}

//
// Update the tracks
//
PlayDetector::TrackStateInfo PlayDetector::update_tracks(
    size_t frame_id,
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes,
    const std::set<size_t>& disregard_tracking_ids) {
  TrackStateInfo ts_info;
  ts_info.track_velocity.reserve(tracking_ids.size());

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
    // We don't record velocities of "disregarded" items (such as a ref),
    // but it's ok to keep trackl of them in our overall tracks, since
    // they may not always be "disregarded".
    PlayerSTrack& track = found_track->second;
    const BBox& tracking_box = tracking_boxes.at(i);
    track.add_position(
        frame_id, tracking_box.anchor_point(), tracking_box.center());
    auto track_velocity = track.velocity() * config_.fps_speed_scale;
    if (!disregard_tracking_ids.count(track_id)) {
      ts_info.track_velocity.emplace(track_id, track_velocity);
      ts_info.cumulative_velocity =
          ts_info.cumulative_velocity + track_velocity;
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

  return ts_info;
}

PlayDetectorResults PlayDetector::forward(
    size_t frame_id,
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes,
    const std::set<size_t>& disregard_tracking_ids) {
  PlayDetectorResults result;
  assert(tracking_ids.size() == tracking_boxes.size());

  //
  // Update the tracks
  //
  TrackStateInfo track_state_info = update_tracks(
      frame_id, tracking_ids, tracking_boxes, disregard_tracking_ids);

  //
  // Ok, now analyze what's going on...
  //
  detect_breakaway(track_state_info);

  return result;
}

void PlayDetector::detect_breakaway(const TrackStateInfo& track_state_info) {
  if (track_state_info.track_velocity.size() <
      kMinPlayersForBreakawayDetection) {
    return;
  }
  std::optional<GroupMovementInfo> group_info =
      get_group_velocity(track_state_info.track_velocity);
  if (!group_info.has_value()) {
    return;
  }
  std::cout << "group velocity: " << group_info->group_velocity << std::endl;
}

std::optional<PlayDetector::GroupMovementInfo> PlayDetector::get_group_velocity(
    const std::unordered_map<size_t, Velocity>& track_velocities) {
  GroupMovementInfo group_info;
  Velocity left_velocity{0.0, 0.0};
  Velocity right_velocity{0.0, 0.0};
  Point leftmost_fast{.x = kLargestPosFloatValue, .y = 0.0};
  Point rightmost_fast{.x = kLargestNegFloatValue, .y = 0.0};
  size_t pos_count = 0, neg_count = 0;
  Velocity pos_sum{0.0, 0.0}, neg_sum{0.0, 0.0};
  for (const auto& track_velocity_item : track_velocities) {
    const Velocity& velocity = track_velocity_item.second;
    const float x_speed = velocity.dx;
    if (x_speed >= config_.min_considered_group_velocity) {
      ++pos_count;
      pos_sum = pos_sum + velocity;
      const size_t track_id = track_velocity_item.first;
      const Point cc = tracks_.at(track_id).center();
      if (cc.x > rightmost_fast.x) {
        rightmost_fast = cc;
      }
    } else if (x_speed <= -config_.min_considered_group_velocity) {
      ++neg_count;
      neg_sum = neg_sum + velocity;
      const size_t track_id = track_velocity_item.first;
      const Point cc = tracks_.at(track_id).center();
      if (cc.x < leftmost_fast.x) {
        leftmost_fast = cc;
      }
    }
  }
  group_info.leftmost_center = leftmost_fast;
  group_info.rightmost_center = rightmost_fast;
  const size_t total_ids = track_velocities.size();
  if (float(pos_count) / total_ids > config_.group_ratio_threshold) {
    // Make sure the opposite isn't also true, or else we need to chose the max
    assert(!(float(neg_count) / total_ids > config_.group_ratio_threshold));
    Velocity average_speed = pos_sum / pos_count;
    group_info.group_velocity = average_speed;
    return group_info;
  } else if (float(neg_count) / total_ids > config_.group_ratio_threshold) {
    // Make sure the opposite isn't also true, or else we need to chose the max
    assert(!(float(pos_count) / total_ids > config_.group_ratio_threshold));
    Velocity average_speed = neg_sum / neg_count;
    group_info.group_velocity = average_speed;
    return group_info;
  }

  return std::nullopt;
}

} // namespace play_tracker
} // namespace hm
