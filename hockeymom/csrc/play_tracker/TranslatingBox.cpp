#include "hockeymom/csrc/play_tracker/TranslatingBox.h"

#include <cassert>
#include <iostream>

namespace hm {
namespace play_tracker {

namespace {
constexpr FloatValue kSpeedDiffDirectionAssumeStoppedMaxRatio = 6.0;
constexpr FloatValue kMaxSpeedDiffDirectionCutRateRatio = 2.0;

} // namespace

TranslatingBox::TranslatingBox(const TranslatingBoxConfig& config)
    : config_(config) {
  if (config_.arena_box.has_value()) {
    gasussian_clamp_lr =
        std::make_pair(config_.arena_box->left, config_.arena_box->right);
  }
}

void TranslatingBox::set_destination(const BBox& dest_box) {
  if (!config_.translation_enabled) {
    return;
  }
  BBox bbox = bounding_box();
  Point center_current = bbox.center();
  Point center_dest = dest_box.center();
  PointDiff total_diff = center_dest - center_current;

  // std::cout << name() << ": total_diff: " << total_diff << std::endl;

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
  } else {
    assert(false);
  }

  if (config_.sticky_translation && !is_nonstop()) {
    //
    // BEGIN Sticky Translation
    //
    const FloatValue diff_magnitude = norm(total_diff);

    // Check if the new center is in a direction opposed to our current
    // velocity
    // const bool changed_direction_x =
    //     (sign(state_.current_speed_x) * sign(total_diff.dx)) < 0;
    // const bool changed_direction_y =
    //     (sign(state_.current_speed_y) * sign(total_diff.dy)) < 0;

    // // Reduce velocities on axes that changed direction
    // const FloatValue velocity_x =
    //     changed_direction_x ? 0.0 : state_.current_speed_x;
    // const FloatValue velocity_y =
    //     changed_direction_y ? 0.0 : state_.current_speed_y;

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
      if (config_.stop_translation_on_dir_change) {
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
      if (config_.stop_translation_on_dir_change) {
        total_diff.dy = 0.0;
      }
    }
  } // end of is_nonstop()

  //adjust_speed(total_diff.dx, total_diff.dy, /*scale_constraints=*/1.0);
  adjust_speed(total_diff.dx, total_diff.dy, /*scale_constraints=*/1.0);
}

const TranslationState& TranslatingBox::get_state() const {
  return state_;
}

PointDiff TranslatingBox::get_proposed_next_position_change() const {
  if (state_.translation_is_frozen) {
    return PointDiff{.dx = zero(), .dy = zero()};
  }
  return PointDiff{.dx = state_.current_speed_x, .dy = state_.current_speed_y};
}

void TranslatingBox::stop_translation_if_out_of_arena() {
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
void TranslatingBox::update_nonstop_delay() {
  if (state_.nonstop_delay != zero()) {
    state_.nonstop_delay_counter += 1;
    if (state_.nonstop_delay_counter > state_.nonstop_delay) {
      state_.nonstop_delay = zero();
      state_.nonstop_delay_counter = zero();
    }
  }
}

const TranslatingBoxConfig& TranslatingBox::get_config() const {
  return config_;
}

void TranslatingBox::on_new_position() {
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
  if (shift_result.was_shifted_x || shift_result.was_shifted_y) {
    set_bbox(shift_result.bbox);
  }
}

bool TranslatingBox::is_nonstop() const {
  return state_.nonstop_delay != 0;
}

void TranslatingBox::clamp_speed(FloatValue scale) {
  state_.current_speed_x = clamp(
      state_.current_speed_x,
      -config_.max_speed_x * scale,
      config_.max_speed_x * scale);
  state_.current_speed_y = clamp(
      state_.current_speed_y,
      -config_.max_speed_y * scale,
      config_.max_speed_y * scale);
}

void TranslatingBox::adjust_speed(
    std::optional<FloatValue> accel_x,
    std::optional<FloatValue> accel_y,
    std::optional<FloatValue> scale_constraints,
    std::optional<IntValue> nonstop_delay) {
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
  } else {
    assert(false);
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
void TranslatingBox::scale_speed(
    std::optional<FloatValue> ratio_x,
    std::optional<FloatValue> ratio_y,
    bool clamp_to_max) {
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

FloatValue TranslatingBox::get_gaussian_y_about_width_center(
    FloatValue x) const {
  // Different than python
  if (!config_.arena_box.has_value()) {
    return 1.0;
  }
  // return 1.0;
  x = clamp(x, gasussian_clamp_lr->first, gasussian_clamp_lr->second);
  const FloatValue center_x = config_.arena_box->center().x;
  if (x < center_x) {
    x = -(center_x - x);
  } else if (x > center_x) {
    x = x - center_x;
  } else {
    return 1.0;
  }
  return 1 - x / center_x;
}

std::tuple<FloatValue, FloatValue> TranslatingBox::
    get_sticky_translation_sizes() const {
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

} // namespace play_tracker
} // namespace hm
