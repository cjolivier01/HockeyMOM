#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"
#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace hm {
namespace play_tracker {

namespace {

// Helper to define a visitor based on lambda expressions
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
} // namespace

class BoundingBox : virtual public IBasicLivingBox {
 public:
  BoundingBox(BBox bbox) : bbox_(bbox.clone()) {}

  void set_bbox(const BBox& bbox) override {
    bbox_ = bbox.clone();
  }

  BBox bounding_box() const override {
    return bbox_.clone();
  }

 protected:
  BBox bbox_;
};

using DrawOptions = std::unordered_map<int, bool>;

template <typename T>
inline T clamp(const T& value, const T& min, const T& max) {
  return std::clamp(value, min, max);
}

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
    set_destination_size(dest_box.width(), dest_box.height());
  }

  const ResizingConfig& resizing_config() const {
    return config_;
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

struct TranslationState {
  FloatValue current_speed_x{0.0};
  FloatValue current_speed_y{0.0};
  bool translation_is_frozen{false};
  // Nonstop stuff
  std::optional<IntValue> nonstop_delay{0};
  IntValue nonstop_delay_counter{0};
};

class TranslatingBox : virtual public IBasicLivingBox {
  static inline constexpr FloatValue kSpeedDiffDirectionAssumeStoppedMaxRatio =
      6.0;
  static inline constexpr FloatValue kMaxSpeedDiffDirectionCutRateRatio = 2.0;

 public:
  TranslatingBox(const TranslatingBoxConfig& config) : config_(config) {
    if (config_.arena_box.has_value()) {
      gasussian_clamp_lr =
          std::make_pair(config_.arena_box->left, config_.arena_box->right);
    }
  }

  void set_destination(const BBox& dest_box) override {
    BBox bbox = bounding_box();
    Point center_current = bbox.center();
    Point center_dest = bbox.center();
    PointDiff total_diff = center_dest - center_current;
    // If both the dest box and our current box are on an edge, we zero-out
    // the magnitude in the direction of that edge so that the size
    // differences of the box don't keep us in the un-stuck mode,
    // even though we can't move anymore in that direction
    if (config_.arena_box.has_value()) {
      // slightly deflated box
      BBox inflated_box = config_.arena_box->inflate(1, 1, -1, -1);
      std::tuple<bool, bool> x_y_on_edge = check_for_box_overshoot(
          bbox,
          inflated_box,
          /*moving_directions=*/total_diff,
          /*epsilon=*/0.1);
      state_.current_speed_x *= !std::get<0>(x_y_on_edge);
      state_.current_speed_y *= !std::get<1>(x_y_on_edge);
      total_diff.dx *= !std::get<0>(x_y_on_edge);
      total_diff.dy *= !std::get<1>(x_y_on_edge);
    }

    if (config_.sticky_translation && !is_nonstop()) {
      //
      // BEGIN Sticky Translation
      //
      const FloatValue diff_magnitude = norm(total_diff);

      // Check if the new center is in a direction opposed to our current
      // velocity
      const bool changed_direction_x =
          (sign(state_.current_speed_x) * sign(total_diff.dx)) < 0;
      const bool changed_direction_y =
          (sign(state_.current_speed_y) * sign(total_diff.dy)) < 0;

      // Reduce velocities on axes that changed direction
      const FloatValue volocity_x =
          changed_direction_x ? 0.0 : state_.current_speed_x;
      const FloatValue velocity_y =
          changed_direction_y ? 0.0 : state_.current_speed_y;

      // See if we are breaking the sticky or unsticky threshold
      const auto sticky_unsticky = get_sticky_translation_sizes();
      const FloatValue sticky_size = std::get<0>(sticky_unsticky);
      const FloatValue unsticky_size = std::get<1>(sticky_unsticky);
      if (!state_.translation_is_frozen) {
        if (diff_magnitude <= sticky_size) {
          state_.translation_is_frozen = true;
          state_.current_speed_x = 0.0;
          state_.current_speed_y = 0.0;
        }
      } else {
        if (diff_magnitude >= unsticky_size) {
          state_.translation_is_frozen = false;
          // Unstick at zero velocity
          state_.current_speed_x = 0.0;
          state_.current_speed_y = 0.0;
        }
      }
      //
      // END Sticky Translation
      //
    }

    if (!is_nonstop()) {
      // If we aren't in a forced state of not applying any
      // constraint/direction-change stops. This is usually because we noticed
      // (externally) a big change and need to get things moving in some
      // corrective way (i.e. follow a breakaway) without any randomness.
      // interfering and stopping it.
      // This is in leiu of making a big jump all at once, which can be
      // (visually) jarring.

      if (different_directions(total_diff.dx, state_.current_speed_x)) {
        if (std::abs(state_.current_speed_x) <
            config_.max_speed_x / kSpeedDiffDirectionAssumeStoppedMaxRatio) {
          state_.current_speed_x = 0.0;
        } else {
          state_.current_speed_x /= kMaxSpeedDiffDirectionCutRateRatio;
        }
        if (config_.stop_on_dir_change) {
          total_diff.dx = 0.0;
        }
      }

      if (different_directions(total_diff.dy, state_.current_speed_y)) {
        if (std::abs(state_.current_speed_y) <
            config_.max_speed_y / kSpeedDiffDirectionAssumeStoppedMaxRatio) {
          state_.current_speed_y = 0.0;
        } else {
          state_.current_speed_y /= kMaxSpeedDiffDirectionCutRateRatio;
        }
        if (config_.stop_on_dir_change) {
          total_diff.dy = 0.0;
        }
      }
    } // end of is_nonstop()

    adjust_speed(total_diff.dx, total_diff.dy);
  }

 protected:
  PointDiff get_proposed_next_position_change() const {
    if (state_.translation_is_frozen) {
      return PointDiff{.dx = zero(), .dy = zero()};
    }
    return PointDiff{
        .dx = state_.current_speed_x, .dy = state_.current_speed_y};
  }

  void stop_translation_if_out_of_arena() {
    if (!config_.arena_box.has_value()) {
      return;
    }
    std::tuple<bool, bool> x_y_on_edge = check_for_box_overshoot(
        bounding_box(),
        config_.arena_box->inflate(1, 1, -1, -1),
        /*moving_directions=*/
        PointDiff{.dx = state_.current_speed_x, .dy = state_.current_speed_y},
        /*epsilon=*/0.1);
    state_.current_speed_x *= !std::get<0>(x_y_on_edge);
    state_.current_speed_y *= !std::get<1>(x_y_on_edge);
  }

  // After new position is set, adjust the nonstop-delay
  void update_nonstop_delay() {
    if (state_.nonstop_delay != zero()) {
      state_.nonstop_delay_counter += 1;
      if (state_.nonstop_delay_counter > state_.nonstop_delay) {
        state_.nonstop_delay = zero();
        state_.nonstop_delay_counter = zero();
      }
    }
  }

  const TranslatingBoxConfig& translating_config() const {
    return config_;
  }

  void on_new_position() {
    if (!config_.arena_box.has_value()) {
      return;
    }
    const ShiftResult shift_result =
        shift_box_to_edge(bounding_box(), *config_.arena_box);
    if (shift_result.was_shifted_x) {
      // We show down X velocity if we went off the edge
      state_.current_speed_x /= kMaxSpeedDiffDirectionCutRateRatio;
    }
    if (shift_result.was_shifted_y) {
      // We show down X velocity if we went off the edge
      state_.current_speed_y /= kMaxSpeedDiffDirectionCutRateRatio;
    }
  }

 private:
  bool is_nonstop() const {
    return state_.nonstop_delay != 0;
  }

  void clamp_speed(FloatValue scale) {
    state_.current_speed_x = clamp(
        state_.current_speed_x,
        -config_.max_speed_x * scale,
        config_.max_speed_x * scale);
    state_.current_speed_y = clamp(
        state_.current_speed_y,
        -config_.max_speed_y * scale,
        config_.max_speed_y * scale);
  }

  void adjust_speed(
      std::optional<FloatValue> accel_x = std::nullopt,
      std::optional<FloatValue> accel_y = std::nullopt,
      std::optional<FloatValue> scale_constraints = std::nullopt,
      std::optional<FloatValue> nonstop_delay = std::nullopt) {
    if (scale_constraints.has_value()) {
      const FloatValue mult = *scale_constraints;
      if (accel_x.has_value()) {
        accel_x = clamp(
            *accel_x, -config_.max_accel_x * mult, config_.max_accel_x * mult);
      }
      if (accel_y.has_value()) {
        accel_y = clamp(
            *accel_y, -config_.max_accel_y * mult, config_.max_accel_y * mult);
      }
    }

    if (accel_x.has_value()) {
      state_.current_speed_x += *accel_x;
    }

    if (accel_y.has_value()) {
      state_.current_speed_y += *accel_y;
    }

    if (scale_constraints.has_value()) {
      clamp_speed(*scale_constraints);
    }
    if (nonstop_delay.has_value()) {
      state_.nonstop_delay = std::move(nonstop_delay);
      state_.nonstop_delay_counter = 0;
    }
  }

  /**
   * Scale the current speed by the given ratio
   */
  void scale_speed(
      std::optional<FloatValue> ratio_x = std::nullopt,
      std::optional<FloatValue> ratio_y = std::nullopt,
      bool clamp_to_max = false) {
    if (ratio_x.has_value()) {
      state_.current_speed_x *= *ratio_x;
    }

    if (ratio_y.has_value()) {
      state_.current_speed_y *= *ratio_y;
    }

    if (clamp_to_max) {
      clamp_speed(1.0);
    }
  }

  FloatValue get_gaussian_y_about_width_center(FloatValue x) const {
    // Different than python
    if (!config_.arena_box.has_value()) {
      return 1.0;
    }
    x = clamp(x, gasussian_clamp_lr->first, gasussian_clamp_lr->second);
    const FloatValue center_x = config_.arena_box->width() / 2;
    if (x < center_x) {
      x = -(center_x - x);
    } else if (x > center_x) {
      x = x - center_x;
    } else {
      return 1.0;
    }
    return 1 - x / center_x;
  }

  std::tuple<FloatValue, FloatValue> get_sticky_translation_sizes() const {
    const BBox bbox = bounding_box();
    const FloatValue gaussian_factor =
        1.0 - get_gaussian_y_about_width_center(bbox.center().x);
    constexpr FloatValue kGaussianMult = 6.0;
    const FloatValue gaussian_add = gaussian_factor * kGaussianMult;

    const FloatValue max_sticky_size =
        config_.max_speed_x * config_.sticky_translation_gaussian_mult +
        gaussian_add;
    FloatValue sticky_size =
        bbox.width() / config_.sticky_size_ratio_to_frame_width;
    sticky_size = std::min(sticky_size, max_sticky_size);

    FloatValue unsticky_size =
        sticky_size * config_.unsticky_translation_size_ratio;
    return std::make_tuple(sticky_size, unsticky_size);
  }

 private:
  TranslatingBoxConfig config_;
  TranslationState state_;

  // Calculated clamp values based upon the given arena box (of any)
  std::optional<std::pair<FloatValue, FloatValue>> gasussian_clamp_lr{
      std::nullopt};
};

class LivingBox : public ILivingBox,
                  public BoundingBox,
                  public ResizingBox,
                  public TranslatingBox {
 public:
  LivingBox(std::string label, BBox bbox, const AllLivingBoxConfig& config);

  void set_destination(
      const std::variant<BBox, const IBasicLivingBox*>& dest) override;

 protected:

  WHDims get_size_scale() const;

  BBox next_position();

  void clamp_to_arena();

  // -IBasicLivingBox
  void set_bbox(const BBox& bbox) override;

  BBox bounding_box() const override;

  void set_destination(const BBox& dest_box) override;
  // IBasicLivingBox-

 private:
  const std::string label_;
  const LivingBoxConfig config_;

  // Flag to show we were size-constrained on the last update
  // (debugging/visualization only)
  bool was_size_contrained_{false};
};

} // namespace play_tracker
} // namespace hm
