#pragma once

#include "hockeymom/csrc/play_tracker/BoxUtils.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace hm {
namespace play_tracker {

/* clang-format off */
/**
 *  _____ ____              _      _      _         _             ____
 * |_   _|  _ \            (_)    | |    (_)       (_)           |  _ \
 *   | | | |_) | __ _  ___  _  ___| |     _ __   __ _ _ __   __ _| |_) | ___ __  __
 *   | | |  _ < / _` |/ __|| |/ __| |    | |\ \ / /| | '_ \ / _` |  _ < / _ \\ \/ /
 *  _| |_| |_) | (_| |\__ \| | (__| |____| | \ V / | | | | | (_| | |_) | (_) |>  <
 * |_____|____/ \__,_||___/|_|\___|______|_|  \_/  |_|_| |_|\__, |____/ \___//_/\_\
 *                                                           __/ |
 *                                                          |___/
 */
/* clang-format on */
struct IBasicLivingBox {
  virtual ~IBasicLivingBox() = default;

  virtual void set_bbox(const BBox& bbox) = 0;

  virtual BBox bounding_box() const = 0;

  virtual void set_destination(const BBox& dest_box) = 0;
};

/**
 *  _____ _      _         _             ____
 * |_   _| |    (_)       (_)           |  _ \
 *   | | | |     _ __   __ _ _ __   __ _| |_) | ___ __  __
 *   | | | |    | |\ \ / /| | '_ \ / _` |  _ < / _ \\ \/ /
 *  _| |_| |____| | \ V / | | | | | (_| | |_) | (_) |>  <
 * |_____|______|_|  \_/  |_|_| |_|\__, |____/ \___//_/\_\
 *                                  __/ |
 *                                 |___/
 */
struct ILivingBox : virtual public IBasicLivingBox {
  virtual void set_destination(
      const std::variant<BBox, const IBasicLivingBox*>& dest) = 0;
};

/* clang-format off */
/**
 *  _____             _       _              _____              __  _
 * |  __ \           (_)     (_)            / ____|            / _|(_)
 * | |__) | ___  ___  _  ____ _ _ __   __ _| |      ___  _ __ | |_  _  __ _
 * |  _  / / _ \/ __|| ||_  /| | '_ \ / _` | |     / _ \| '_ \|  _|| |/ _` |
 * | | \ \|  __/\__ \| | / / | | | | | (_| | |____| (_) | | | | |  | | (_| |
 * |_|  \_\\___||___/|_|/___||_|_| |_|\__, |\_____|\___/|_| |_|_|  |_|\__, |
 *                                     __/ |                           __/ |
 *                                    |___/                           |___/
 */
/* clang-format on */
struct ResizingConfig {
  FloatValue max_speed_w{0.0};
  FloatValue max_speed_h{0.0};
  FloatValue max_accel_w{0.0};
  FloatValue max_accel_h{0.0};
  FloatValue min_width{0.0};
  FloatValue min_height{0.0};
  FloatValue max_width{0.0};
  FloatValue max_height{0.0};
  bool stop_on_dir_change{true};
  bool sticky_sizing{false};
  //
  // Sticky sizing thresholds
  //
  // Threshold to grow width (ratio of bbox)
  FloatValue size_ratio_thresh_grow_dw{0.05};
  // Threshold to grow height (ratio of bbox)
  FloatValue size_ratio_thresh_grow_dh{0.1};
  // Threshold to shrink width (ratio of bbox)
  FloatValue size_ratio_thresh_shrink_dw{0.08};
  // Threshold to shrink height (ratio of bbox)
  FloatValue size_ratio_thresh_shrink_dh{0.1};
};

/* clang-format off */
/**
 *  _______                     _       _   _              _____              __  _
 * |__   __|                   | |     | | (_)            / ____|            / _|(_)
 *    | |_ __  __ _ _ __   ___ | | __ _| |_ _  ___  _ __ | |      ___  _ __ | |_  _  __ _
 *    | | '__|/ _` | '_ \ / __|| |/ _` | __| |/ _ \| '_ \| |     / _ \| '_ \|  _|| |/ _` |
 *    | | |  | (_| | | | |\__ \| | (_| | |_| | (_) | | | | |____| (_) | | | | |  | | (_| |
 *    |_|_|   \__,_|_| |_||___/|_|\__,_|\__|_|\___/|_| |_|\_____|\___/|_| |_|_|  |_|\__, |
 *                                                                                   __/ |
 *                                                                                  |___/
 */
/* clang-format on */
struct TranslatingBoxConfig {
  FloatValue max_speed_x{0.0};
  FloatValue max_speed_y{0.0};
  FloatValue max_accel_x{0.0};
  FloatValue max_accel_y{0.0};
  bool stop_on_dir_change{true};
  std::optional<BBox> arena_box{std::nullopt};
  std::optional<FloatValue> fixed_aspect_ratio{std::nullopt};
  bool clamp_scaled_input_box{true};
  // Sticky Sizing
  bool sticky_translation{false};
  FloatValue sticky_size_ratio_to_frame_width{10.0};
  FloatValue sticky_translation_gaussian_mult{5.0};
  FloatValue unsticky_translation_size_ratio{0.75};
};

struct LivingBoxConfig {
  FloatValue scale_dest_width{1.0};
  FloatValue scale_dest_height{1.0};
  std::optional<FloatValue> fixed_aspect_ratio{std::nullopt};
};

struct AllLivingBoxConfig : public ResizingConfig,
                            public TranslatingBoxConfig,
                            public LivingBoxConfig {};

std::unique_ptr<ILivingBox> create_live_box(
    std::string label,
    const BBox& bbox,
    const AllLivingBoxConfig* config = nullptr);

} // namespace play_tracker
} // namespace hm
