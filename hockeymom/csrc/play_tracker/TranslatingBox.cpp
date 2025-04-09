#include "hockeymom/csrc/play_tracker/TranslatingBox.h"

#include <cassert>
#include <iomanip>
#include <ios>
#include <iostream>

#include <unistd.h>

namespace hm {
namespace play_tracker {

namespace {
constexpr FloatValue kSpeedDiffDirectionAssumeStoppedMaxRatio = 6.0;
constexpr FloatValue kMaxSpeedDiffDirectionCutRateRatio = 2.0;
constexpr FloatValue kDestinationDistanceToArenaWidthRatioToIgnoreScalingSpeed =
    1.0f / 3;

// This would actually be the tilt amount we configure for final video
// presenation
constexpr FloatValue kDegreesEdgePerspecive = 20.0;
// const FloatValue kSineEdgePerspecive = std::sin(kDegreesEdgePerspecive);

constexpr FloatValue kEpsilon = 1e-4f;
constexpr FloatValue kSmallTestDistance = 10.0f;

bool is_close(
    const FloatValue& f1,
    const FloatValue& f2,
    const FloatValue epsilon = kEpsilon) {
  return std::abs(f2 - f1) < epsilon;
}

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

  FloatValue x_gaussian = 1.0;
  if (config_.dynamic_acceleration_scaling) {
    assert(config_.sticky_translation);
    static size_t test_pass_counter = 0;
    if (!test_pass_counter++) {
      test_arena_edge_position_scale();
    }
    // Only do this if we aren't super off so that massive movements are still
    // possible in desperate situations
    // TODO: We can cache this computation
    if (total_diff.dx < config_.arena_box->width() *
            kDestinationDistanceToArenaWidthRatioToIgnoreScalingSpeed) {
      x_gaussian = 1.0 - get_arena_edge_position_scale();
    } else {
      static size_t wayoff_count = 0;
      std::cout << ++wayoff_count
                << ": We are way off, ignoring any position scale\n";
    }
    //   x_gaussian =
    //       1.0 - get_gaussian_ratio(total_diff.dx,
    //       config_.arena_box->width());
  }

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

  // adjust_speed(total_diff.dx, total_diff.dy, /*scale_constraints=*/1.0);
  adjust_speed(total_diff.dx, total_diff.dy, /*scale_constraints=*/x_gaussian);

  // static int ctr = 0;
  // if (ctr++ % 5) {
  //   std::cout << "total_diff=" << total_diff.dx << ", " << total_diff.dy
  //             << ", edge scale: " << get_arena_edge_position_scale() << "\n";
  // }
  if (config_.sticky_translation) {
    get_arena_edge_position_scale();
  }
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
    // if (config_.dynamic_acceleration_scaling) {
    //   std::cout << "Scale max accel: " << mult << std::endl;
    // }
    if (accel_x.has_value()) {
      accel_x = clamp(
          *accel_x, -config_.max_accel_x * mult, config_.max_accel_x * mult);
    }
    if (accel_y.has_value()) {
      accel_y = clamp(
          *accel_y, -config_.max_accel_y * mult, config_.max_accel_y * mult);
    }
  } else {
    assert(false); // always, for now
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
  // Just simple linear scaling
  return 1 - x / center_x;
}

FloatValue TranslatingBox::get_gaussian_ratio(
    FloatValue position,
    FloatValue overall_length) const {
  position = std::abs(position);
  if (position > overall_length) {
    position = overall_length;
  }
  FloatValue gaussian_length =
      (gasussian_clamp_lr->second - gasussian_clamp_lr->first) / 2;
  assert(gaussian_length > 0.0);
  position *= overall_length / gaussian_length;
  // We just sample the right side of the gaussian curve
  position += config_.arena_box->center().x;
  FloatValue gaussian = get_gaussian_y_about_width_center(position);
  // std::cout << "position=" << position << ", overall_length=" <<
  // overall_length
  //           << ", gaussian=" << gaussian << std::endl;
  assert(gaussian >= 0 && gaussian <= 1);
  return gaussian;
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

static FloatValue adjusted_horizontal_distance_from_edge(
    FloatValue x,
    FloatValue y,
    const BBox& arena_box,
    const FloatValue field_edge_veritcal_angle) {
  const FloatValue half_height = arena_box.height() / 2;
  const FloatValue half_width = arena_box.width() / 2;
  // On the edges, the perspective will have only the top half or so with
  // ice/field, with the center of the field warping down to the bottom in the
  // middle
  FloatValue percent_y =
      (std::min(half_height, y) - arena_box.top) / half_height;
  FloatValue max_x_adjusted_distance =
      std::sin(field_edge_veritcal_angle) * half_height;
  FloatValue x_adjusted_distance = max_x_adjusted_distance * (1.0 - percent_y);

#if 1
  FloatValue x_adjusted_width = half_width - x_adjusted_distance;
  assert(x_adjusted_width > 0);
  // Scale back x_adjusted_distance based on how close to center
  FloatValue dist_from_center = std::abs(x_adjusted_width - x);
  FloatValue dist_from_center_ratio = dist_from_center / x_adjusted_width;
  x_adjusted_distance *= dist_from_center_ratio;
#endif

  FloatValue adjusted_x_distance_from_edge =
      std::max(x - x_adjusted_distance, 0.0f);
  return std::min(adjusted_x_distance_from_edge, half_width);
}

void TranslatingBox::test_arena_edge_position_scale() {
  const BBox arena_box = *config_.arena_box;
  const Point arena_center = arena_box.center();
  FloatValue full_left =
      get_arena_edge_position_scale(Point{.x = 0, .y = arena_center.y});
  FloatValue arena_left = get_arena_edge_position_scale(
      Point{.x = arena_box.left, .y = arena_center.y});
  FloatValue full_center = get_arena_edge_position_scale(
      Point{.x = arena_center.x, .y = arena_center.y});
  FloatValue full_right = get_arena_edge_position_scale(
      Point{.x = arena_box.right * 2, .y = arena_center.y});
  FloatValue arena_right = get_arena_edge_position_scale(
      Point{.x = arena_box.right, .y = arena_center.y});
  // The edge cases, far edges = 1, dead center = 0
  assert(is_close(full_left, 1.0f));
  assert(is_close(full_left, arena_left));
  assert(is_close(full_right, 1.0f));
  assert(is_close(full_right, arena_right));
  assert(is_close(full_center, 0.0f));

  // Now check that the calculation is continuous near those edge points
  // (doesn't jump to some crazy value)
  FloatValue far_left = get_arena_edge_position_scale(
      Point{.x = arena_box.left + kSmallTestDistance, .y = arena_center.y});
  FloatValue far_right = get_arena_edge_position_scale(
      Point{.x = arena_box.right - kSmallTestDistance, .y = arena_center.y});
  FloatValue left_of_center = get_arena_edge_position_scale(
      Point{.x = arena_center.x - kSmallTestDistance, .y = arena_center.y});
  FloatValue right_of_center = get_arena_edge_position_scale(
      Point{.x = arena_center.x + kSmallTestDistance, .y = arena_center.y});

  // std::cout << far_left << ", " << left_of_center << ", " << right_of_center
  //           << ", " << far_right << std::endl;

  assert(is_close(far_left, far_right));
  assert(is_close(left_of_center, right_of_center));
  assert(far_left > left_of_center && far_right > right_of_center);
  // Don't want to restrict/define the algorithm in this test, so just make sure
  // it's some reasonable value.
  assert(std::fabs(1.0 - far_left) < kSmallTestDistance / 10);
  assert(left_of_center < kSmallTestDistance / 10);

  // Now check differenced in Y
  FloatValue adjusted_x_last = 0;
  for (FloatValue y = arena_box.bottom; y >= 0.0f; y -= 100) {
    FloatValue left_x = arena_box.center().x - arena_box.left / 2;
    FloatValue adjusted_x = adjusted_horizontal_distance_from_edge(
        left_x, y, arena_box, kDegreesEdgePerspecive);
    std::cout << adjusted_x << ", ";
    if (adjusted_x_last) {
      assert(adjusted_x <= adjusted_x_last);
    }
    adjusted_x_last = adjusted_x;
  }
  std::cout << std::endl;
  usleep(0);
}

FloatValue TranslatingBox::get_arena_edge_position_scale(
    const Point& pt) const {
  const BBox& arena_box = *config_.arena_box;
  const FloatValue arena_center_x = arena_box.center().x;
  if (pt.x == arena_center_x) {
    return 0.0;
  }
  FloatValue x_eff;
  if (pt.x < arena_center_x) {
    x_eff = std::max(0.0f, pt.x - arena_box.left);
  } else {
    x_eff = std::max(0.0f, arena_box.right - pt.x);
  }
  FloatValue x_eff_adjusted = adjusted_horizontal_distance_from_edge(
      x_eff, pt.y, arena_box, kDegreesEdgePerspecive);
  const FloatValue half_arena_width = arena_box.width() / 2;
  FloatValue ratio = 1.0f - (x_eff_adjusted / half_arena_width);
  return ratio;
}

FloatValue TranslatingBox::get_arena_edge_position_scale() const {
  assert(config_.arena_box.has_value());
  const BBox bbox = bounding_box();
#if 1
  FloatValue ratio = get_arena_edge_position_scale(bbox.center());
#else
  const BBox& arena_box = *config_.arena_box;

  FloatValue left_dist_eff = adjusted_horizontal_distance_from_edge(
      std::max(0.0f, bbox.center().x - arena_box.left),
      bbox.center().y,
      arena_box);

  const FloatValue half_arena_width = arena_box.width() / 2;
  assert(left_dist_eff >= 0 && left_dist_eff <= half_arena_width);

  FloatValue right_dist_eff = adjusted_horizontal_distance_from_edge(
      std::max(0.0f, arena_box.right - bbox.center().x),
      bbox.center().y,
      arena_box);

  assert(right_dist_eff >= 0 && right_dist_eff <= half_arena_width);

  if (left_dist_eff == 0.0) {
    usleep(0);
  }
  if (right_dist_eff == 0.0) {
    usleep(0);
  }
  FloatValue left_side_percent_x = 1.0f - left_dist_eff / half_arena_width;
  FloatValue right_side_percent_x = 1.0f - right_dist_eff / half_arena_width;

  FloatValue ratio = std::max(left_side_percent_x, right_side_percent_x);

  FloatValue new_calc = get_arena_edge_position_scale(bbox.center());
  std::cout << ratio << ", " << new_calc << std::endl;

  // if (bbox.center().x < arena_box.center().x) {
  //   ratio = left_side_percent_x;
  // } else if (bbox.center().x > arena_box.center().x) {
  //   ratio = right_side_percent_x;
  // }
#endif
  std::cout << "X Scale Ratio: " << ratio << "\n";
  return ratio;
}

} // namespace play_tracker
} // namespace hm
