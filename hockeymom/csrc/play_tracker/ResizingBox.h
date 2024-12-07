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
  ResizingBox(const ResizingConfig& config) : config_(config) {}

  void set_destination(const BBox& dest_box) override {
    if (!config_.resizing_enabled) {
      return;
    }
    set_destination_size(dest_box.width(), dest_box.height());
  }

  const ResizingConfig& get_config() const {
    return config_;
  }

  const ResizingState& get_state() const {
    return state_;
  }

 protected:
  SizeDiff get_proposed_next_size_change() const {
    if (state_.size_is_frozen) {
      return SizeDiff{.dw = zero(), .dh = zero()};
    }
    return SizeDiff{
        .dw = state_.current_speed_w / 2, .dh = state_.current_speed_h / 2};
  }

  WHDims get_min_allowed_width_height() const {
    return WHDims{.width = config_.min_width, .height = config_.min_height};
  }

  // WHDims get_max_allowed_width_height() const {
  //   return WHDims{.width = config_.max_width, .height = config_.max_height};
  // }

  void clamp_size_scaled() {
    const BBox bbox = bounding_box();
    float w = bbox.width();
    float h = bbox.height();
    float wscale = zero(), hscale = zero();
    if (w > config_.max_width) {
      wscale = config_.max_width / w;
    }
    if (h > config_.max_height) {
      hscale = config_.max_height / h;
    }
    float final_scale = std::max(wscale, hscale);
    if (!isZero(final_scale)) {
      w *= final_scale;
      h *= final_scale;
      set_bbox(BBox(bbox.center(), WHDims{.width = w, .height = h}));
    }
  }

 private:
  void set_destination_size(
      FloatValue dest_width,
      FloatValue dest_height,
      bool prioritize_width_thresh = true) {
    BBox bbox = bounding_box();
    auto current_w = bbox.width();
    auto current_h = bbox.height();

    auto dw = dest_width - current_w;
    auto dh = dest_height - current_h;

    if (config_.sticky_sizing) {
      //
      // Begin size threshholding stuff
      //
      GrowShrink resize_rates = get_grow_shrink_wh(bbox);

      // dw
      bool dw_thresh = dw < 0 && dw < -resize_rates.shrink_width;
      const bool want_bigger_w = dw > 0 && dw > resize_rates.grow_width;
      dw_thresh |= want_bigger_w;
      if (!dw_thresh) {
        dw = 0.0f;
      }

      // dh
      bool dh_thresh = dh < 0 && dh < -resize_rates.shring_height;
      const bool want_bigger_h = dh > 0 && dh > resize_rates.grow_height;
      dh_thresh |= want_bigger_h;
      if (!dh_thresh) {
        dh = 0.0f;
      }

      bool both_thresh = dw_thresh && dh_thresh;
      if (prioritize_width_thresh) {
        both_thresh |= dw_thresh;
      }
      const bool any_thresh = dw_thresh || dh_thresh;
      const bool want_bigger = (want_bigger_w || want_bigger_h) && any_thresh;
      state_.size_is_frozen |= !(both_thresh || want_bigger);
      //
      // End size threshholding stuff
      //
    }

    constexpr FloatValue kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio =
        6.0;
    constexpr FloatValue kMaxWidthHeightDiffDirectionCutRateRatio = 2.0;
    if (different_directions(dw, state_.current_speed_w)) {
      // The desired change is in the opposire direction of the current widening
      if (std::abs(state_.current_speed_w) <
          (config_.max_speed_w /
           kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio)) {
        // It's small enough, so just stop the velocity in the opposite
        // direction of the desired change
        state_.current_speed_w = 0;
      } else {
        state_.current_speed_w /= kMaxWidthHeightDiffDirectionCutRateRatio;
      }
    }
    if (different_directions(dh, state_.current_speed_h)) {
      // The desired change is in the opposire direction of the current
      // heightening
      if (std::abs(state_.current_speed_h) <
          (config_.max_speed_h /
           kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio)) {
        // It's small enough, so just stop the velocity in the opposite
        // direction of the desired change
        state_.current_speed_h = 0;
      } else {
        state_.current_speed_h /= kMaxWidthHeightDiffDirectionCutRateRatio;
      }
    }

    adjust_size(/*accel_w=*/dw, /*accel_h=*/dh);
  }

  void adjust_size(
      FloatValue accel_w,
      FloatValue accel_h,
      bool use_constraints = true) {
    if (state_.size_is_frozen) {
      return;
    }

    if (use_constraints) {
      constexpr FloatValue kResizeLargerScaleDifference = 2.0;

      // Growing is allowed at a higher rate than shrinking
      const FloatValue max_accel_w = accel_w > 0
          ? (config_.max_accel_w * kResizeLargerScaleDifference)
          : config_.max_accel_w;
      const FloatValue max_accel_h = accel_h > 0
          ? (config_.max_accel_h * kResizeLargerScaleDifference)
          : config_.max_accel_h;

      accel_w = clamp(accel_w, -max_accel_w, max_accel_w);
      accel_h = clamp(accel_h, -max_accel_h, max_accel_h);
    }

    state_.current_speed_w += accel_w;
    state_.current_speed_h += accel_h;

    if (use_constraints) {
      clamp_resizing();
    }
  }

  void clamp_resizing() {
    state_.current_speed_w = clamp(
        state_.current_speed_w, -config_.max_speed_w, config_.max_speed_w);
    state_.current_speed_h = clamp(
        state_.current_speed_h, -config_.max_speed_h, config_.max_speed_h);
  }

  struct GrowShrink {
    FloatValue grow_width, grow_height, shrink_width, shring_height;
  };

  GrowShrink get_grow_shrink_wh(const BBox& bbox) const {
    auto ww = bbox.width(), hh = bbox.height();
    return GrowShrink{
        .grow_width = ww * config_.size_ratio_thresh_grow_dw,
        .grow_height = hh * config_.size_ratio_thresh_grow_dh,
        .shrink_width = ww * config_.size_ratio_thresh_shrink_dw,
        .shring_height = hh * config_.size_ratio_thresh_shrink_dh,
    };
  }

  const ResizingConfig config_;
  ResizingState state_;
};


} // namespace play_tracker
} // namespace hm
