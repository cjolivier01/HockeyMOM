#include "hockeymom/csrc/play_tracker/ResizingBox.h"

#include <cassert>
#include <iostream>

#include <unistd.h>

namespace hm {
namespace play_tracker {

namespace {
constexpr FloatValue kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio = 6.0;
constexpr FloatValue kMaxWidthHeightDiffDirectionCutRateRatio = 2.0;
constexpr FloatValue kResizeLargerScaleDifference = 2.0;
} // namespace

ResizingBox::ResizingBox(const ResizingConfig& config) : config_(config) {}

void ResizingBox::set_destination(const BBox& dest_box) {
  if (!config_.resizing_enabled) {
    return;
  }
  set_destination_size(dest_box.width(), dest_box.height());
}

const ResizingConfig& ResizingBox::get_config() const {
  return config_;
}

const ResizingState& ResizingBox::get_state() const {
  return state_;
}

SizeDiff ResizingBox::get_proposed_next_size_change() const {
  // std::cout << name() << ": current_speed_w = " << state_.current_speed_w
  //           << ", current_speed_h = " << state_.current_speed_h << std::endl;
  if (state_.size_is_frozen) {
    return SizeDiff{.dw = zero(), .dh = zero()};
  }
  return SizeDiff{
      .dw = state_.current_speed_w / 2, .dh = state_.current_speed_h / 2};
}

WHDims ResizingBox::get_min_allowed_width_height() const {
  return WHDims{.width = config_.min_width, .height = config_.min_height};
}

void ResizingBox::clamp_size_scaled() {
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
    float ar = w / h;
    // Fix some tiny float divergences here
    constexpr float kEplison = 0.0001;
    if (w > config_.max_width) {
      assert(w - config_.max_width < kEplison);
      w = config_.max_width;
      // h = w / ar;
    }
    if (h > config_.max_height) {
      assert(h - config_.max_height < kEplison);
      h = config_.max_height;
    }
    BBox new_box = BBox(bbox.center(), WHDims{.width = w, .height = h});
    set_bbox(BBox(bbox.center(), WHDims{.width = w, .height = h}));
  }
}

void ResizingBox::set_destination_size(
    FloatValue dest_width,
    FloatValue dest_height,
    bool prioritize_width_thresh) {
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
    bool dh_thresh = dh < 0 && dh < -resize_rates.shrink_height;
    const bool want_bigger_h = dh > 0 && dh > resize_rates.grow_height;
    dh_thresh |= want_bigger_h;
    if (!dh_thresh) {
      dh = 0.0f;
    }

    bool freeze_size = !dh_thresh && !dh_thresh;

    bool both_thresh = dw_thresh && dh_thresh;
    if (prioritize_width_thresh) {
      both_thresh |= dw_thresh;
    }
    const bool any_thresh = dw_thresh || dh_thresh;
    const bool want_bigger = (want_bigger_w || want_bigger_h) && any_thresh;

    // bool was_frozen = state_.size_is_frozen;
    state_.size_is_frozen = freeze_size || !(both_thresh || want_bigger);
    // if (was_frozen && !state_.size_is_frozen) {
    //   std::cout << "Unfreezing size" << std::endl;
    // }
    // if (state_.size_is_frozen) {
    //   usleep(0);
    // }

    //
    // End size threshholding stuff
    //
  }

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

void ResizingBox::adjust_size(
    FloatValue accel_w,
    FloatValue accel_h,
    bool use_constraints) {
  if (state_.size_is_frozen) {
    return;
  }

  if (use_constraints) {
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

void ResizingBox::clamp_resizing() {
  state_.current_speed_w =
      clamp(state_.current_speed_w, -config_.max_speed_w, config_.max_speed_w);
  state_.current_speed_h =
      clamp(state_.current_speed_h, -config_.max_speed_h, config_.max_speed_h);
}

GrowShrink ResizingBox::get_grow_shrink_wh(const BBox& bbox) const {
  auto ww = bbox.width(), hh = bbox.height();
  return GrowShrink{
      .grow_width = ww * config_.size_ratio_thresh_grow_dw,
      .grow_height = hh * config_.size_ratio_thresh_grow_dh,
      .shrink_width = ww * config_.size_ratio_thresh_shrink_dw,
      .shrink_height = hh * config_.size_ratio_thresh_shrink_dh,
  };
}

} // namespace play_tracker
} // namespace hm
