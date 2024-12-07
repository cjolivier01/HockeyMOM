#pragma once

#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace hm {
namespace play_tracker {

struct ResizingState {
  bool size_is_frozen{true};
  FloatValue current_speed_w{0.0};
  FloatValue current_speed_h{0.0};
};

class ResizingBox : virtual public IBasicLivingBox {
 public:
  ResizingBox(ResizingBox&&) = delete;
  ResizingBox(const ResizingConfig& config);

  void set_destination(const BBox& dest_box) override;

  const ResizingState& get_state() const;

 protected:
  const ResizingConfig& get_config() const;

  SizeDiff get_proposed_next_size_change() const;

  WHDims get_min_allowed_width_height() const;

  void clamp_size_scaled();

 private:
  void set_destination_size(
      FloatValue dest_width,
      FloatValue dest_height,
      bool prioritize_width_thresh = true);

  void adjust_size(
      FloatValue accel_w,
      FloatValue accel_h,
      bool use_constraints = true);

  void clamp_resizing();

  struct GrowShrink {
    FloatValue grow_width, grow_height, shrink_width, shring_height;
  };

  GrowShrink get_grow_shrink_wh(const BBox& bbox) const;

  const ResizingConfig config_;
  ResizingState state_;
};

} // namespace play_tracker
} // namespace hm
