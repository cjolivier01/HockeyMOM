#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"
#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <cassert>

namespace hm {
namespace play_tracker {

struct TranslationState {
  FloatValue current_speed_x{0.0};
  FloatValue current_speed_y{0.0};
  bool translation_is_frozen{false};
  // Nonstop stuff
  std::optional<IntValue> nonstop_delay{0};
  IntValue nonstop_delay_counter{0};
};

class TranslatingBox : virtual public IBasicLivingBox {
 public:
  TranslatingBox(const TranslatingBoxConfig& config);

  void set_destination(const BBox& dest_box) override;

  const TranslationState& get_state() const;

  const TranslatingBoxConfig& get_config() const;

  void adjust_speed(
      std::optional<FloatValue> accel_x = std::nullopt,
      std::optional<FloatValue> accel_y = std::nullopt,
      std::optional<FloatValue> scale_constraints = std::nullopt,
      std::optional<IntValue> nonstop_delay = std::nullopt);

  /**
   * Scale the current speed by the given ratio
   */
  void scale_speed(
      std::optional<FloatValue> ratio_x = std::nullopt,
      std::optional<FloatValue> ratio_y = std::nullopt,
      bool clamp_to_max = false);

  std::tuple<FloatValue, FloatValue> get_sticky_translation_sizes() const;

  // From the current position and size, get the ratio of closeness to the left or right
  // edge, where 1.0 means on the edge and 0.0 means in the middle.
  // The size of the box may be part of the consideration, for example a very large
  // box will cover more horizontal ground and thus be less "completely on the edge",
  // so some of the result will be a heuristic wrt size and position.
  FloatValue get_arena_edge_position_scale() const;
  FloatValue get_arena_edge_position_scale(const Point& pt) const;

 protected:
  PointDiff get_proposed_next_position_change() const;

  void stop_translation_if_out_of_arena();

  // After new position is set, adjust the nonstop-delay
  void update_nonstop_delay();

  void on_new_position();

 private:
  void test_arena_edge_position_scale();

  bool is_nonstop() const;

  void clamp_speed(FloatValue scale);

  FloatValue get_gaussian_y_about_width_center(FloatValue x) const;
  // FloatValue get_gaussian_ratio(FloatValue position, FloatValue overall_length) const;

 private:
  TranslatingBoxConfig config_;
  TranslationState state_;

  // Calculated clamp values based upon the given arena box (of any)
  std::optional<std::pair<FloatValue, FloatValue>> gasussian_clamp_lr{
      std::nullopt};
};

} // namespace play_tracker
} // namespace hm
