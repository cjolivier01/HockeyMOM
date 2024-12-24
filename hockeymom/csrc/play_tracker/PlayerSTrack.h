#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"

#include <cassert>
#include <deque>

namespace hm {
namespace play_tracker {

class PlayerSTrack {
  static inline constexpr size_t kMinPositionDiffsToCalculateVelocity = 2;

 public:
  PlayerSTrack(
      size_t max_positions,
      size_t max_velocity_positions,
      size_t frame_step = 1)
      : max_positions_(max_positions),
        max_velocity_positions_(max_velocity_positions),
        frame_step_(frame_step) {
    assert(max_velocity_positions_ <= max_positions_);
  }

  void add_position(size_t frame_id, const Point& position) {
    // If it was a lost frame, start everything over
    if (last_frame_id_ + frame_step_ != frame_id) {
      reset();
    } else {
      assert(!frame_id || frame_id - frame_step_ == last_frame_id_);
    }

    if (positions_.size() >= max_velocity_positions_) {
      // TODO: is this correct?
      sum_position_diffs_ = sum_position_diffs_ - position_diffs_.front();
    }
    if (positions_.size() >= max_positions_) {
      positions_.pop_front();
    }
    if (!positions_.empty()) {
      auto last_position = positions_.back();
      auto position_diff = PointDiff{
          .dx = position.x - last_position.x,
          .dy = position.y - last_position.y};
      position_diffs_.push_back(position_diff);
      sum_position_diffs_ = sum_position_diffs_ + position_diff;
    }
    positions_.push_back(position);
    last_frame_id_ = frame_id;
  }

  size_t age(size_t frame_id) const {
    assert(frame_id >= last_frame_id_);
    return frame_id - last_frame_id_;
  }

  void reset() {
    sum_position_diffs_ = PointDiff{0.0f, 0.0f};
    positions_.clear();
  }

  PointDiff velocity() const {
    if (position_diffs_.size() < kMinPositionDiffsToCalculateVelocity) {
      return PointDiff{0.0f, 0.0f};
    }
    assert(position_diffs_.size() <= max_velocity_positions_);
    // We only calculate velocity for the last max_velocity_positions_
    const float velocity_count = position_diffs_.size();
    return PointDiff{
        .dx = sum_position_diffs_.dx / velocity_count,
        .dy = sum_position_diffs_.dy / velocity_count};
  }

 private:
  const size_t max_positions_;
  const size_t max_velocity_positions_;
  const size_t frame_step_;
  std::deque<Point> positions_;
  std::deque<PointDiff> position_diffs_;
  PointDiff sum_position_diffs_{0.0f, 0.0f};
  size_t last_frame_id_{0};
};

} // namespace play_tracker
} // namespace hm
