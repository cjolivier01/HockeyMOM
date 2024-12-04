#include <ATen/ATen.h>
#include <torch/torch.h>

#include "hockeymom/csrc/play_tracker/MovingBox.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

// using namespace cv;
using namespace torch;

namespace {

// using BBox = torch::Tensor;
// using FloatValue = torch::Tensor;
using FloatValue = float;

inline float sign(const float value) {
  if (value > 0) {
    return 1.0f;
  } else if (value < 0) {
    return -1.0f;
  } else {
    return 0.0f;
  }
}

inline bool different_directions(const float v1, const float v2) {
  return sign(v1) * sign(v2) < 0.0;
}
} // namespace

struct WHDims {
  const FloatValue width;
  const FloatValue height;
};

struct PointDiff {
  const FloatValue dx;
  const FloatValue dy;
};

struct Point {
  FloatValue x;
  FloatValue y;

  PointDiff operator-(const Point& pt) const {
    return PointDiff{.dx = x - pt.x, .dy = y - pt.y};
  }
};

struct BBox {
  BBox() = default;
  BBox(float l, float t, float r, float b)
      : left(l), top(t), right(r), bottom(b) {}
  BBox(const Point& center, const WHDims& dims) {
    auto half_w = dims.width / 2;
    auto half_h = dims.height / 2;
    left = center.x - half_w;
    right = center.x + half_w;
    top = center.y - half_h;
    bottom = center.y + half_h;
  }
  constexpr float width() const {
    assert(right >= left);
    return right - left;
  }
  constexpr float height() const {
    assert(bottom >= top);
    return bottom - top;
  }
  BBox clone() const {
    return *this;
  }
  Point center() const {
    return Point{.x = (right - left) / 2, .y = (bottom - top) / 2};
  }
  BBox make_scaled(float scale_width, float scale_height) const {
    return BBox(
        center(),
        WHDims{
            .width = width() * scale_width, .height = height() * scale_height});
  }
  BBox inflate(float dleft, float dtop, float dright, float dbottom) const {
    return BBox(left + dleft, top + dtop, right + dright, bottom + dbottom);
  }
  // The four bbox values
  float left{0.0};
  float top{0.0};
  float right{0.0};
  float bottom{0.0};
};

std::tuple<bool, bool, bool, bool> is_box_edge_on_or_outside_other_box_edge(
    const BBox& box,
    const BBox& bounding_box) {
  return std::make_tuple(
      box.left <= bounding_box.left,
      box.top <= bounding_box.top,
      box.right >= bounding_box.right,
      box.bottom >= bounding_box.bottom);
}

std::tuple<bool, bool> check_for_box_overshoot(
    const BBox& box,
    const BBox& bounding_box,
    const PointDiff& moving_directions,
    float epsilon = 0.01) {
  const auto any_on_edge =
      is_box_edge_on_or_outside_other_box_edge(box, bounding_box);

  const bool left_on_edge =
      std::get<0>(any_on_edge) && moving_directions.dx < epsilon;
  const bool right_on_edge =
      std::get<2>(any_on_edge) && moving_directions.dx > -epsilon;
  const bool x_on_edge = left_on_edge || right_on_edge;

  const bool top_on_edge =
      std::get<1>(any_on_edge) && moving_directions.dy < epsilon;
  const bool bottom_on_edge =
      std::get<3>(any_on_edge) && moving_directions.dy > -epsilon;
  const bool y_on_edge = left_on_edge || right_on_edge;

  return std::make_tuple(x_on_edge, y_on_edge);
}

struct IBasicBox {
  virtual ~IBasicBox() = default;

  virtual void set_bbox(const BBox& bbox) = 0;

  virtual BBox bounding_box() const = 0;

  virtual void set_destination(const BBox& dest_box) = 0;

  static constexpr float one() {
    return 1.0;
  }

  static constexpr float zero() {
    return 0.0;
  }

  static constexpr int64_t zero_int() {
    return 0;
  }
};

class BasicBox : public IBasicBox {
 public:
  BasicBox(BBox bbox) : bbox_(bbox.clone()) {}

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
  std::clamp(value, min, max);
}

struct ResizingBoxConfig {
  float max_speed_w{0.0};
  float max_speed_h{0.0};
  float max_accel_w{0.0};
  float max_accel_h{0.0};
  float min_width{0.0};
  float min_height{0.0};
  float max_width{0.0};
  float max_height{0.0};
  bool stop_on_dir_change{true};
  bool sticky_sizing{false};
  //
  // Sticky sizing thresholds
  //
  // Threshold to grow width (ratio of bbox)
  const float size_ratio_thresh_grow_dw{0.05};
  // Threshold to grow height (ratio of bbox)
  const float size_ratio_thresh_grow_dh{0.1};
  // Threshold to shrink width (ratio of bbox)
  const float size_ratio_thresh_shrink_dw{0.08};
  // Threshold to shrink height (ratio of bbox)
  const float size_ratio_thresh_shrink_dh{0.1};
};

struct ResizingState {
  bool size_is_frozen{true};
  float current_speed_w{0.0};
  float current_speed_h{0.0};
};

class ResizingBox : public IBasicBox {
 public:
  ResizingBox(ResizingBox&&) = delete;
  ResizingBox(const ResizingBoxConfig& config) : config_(config) {}

  virtual void draw(at::Tensor& img, bool draw_thresholds = true) {
    // cv::Rect bbox_rect(
    //     static_cast<int>(bbox_[0].item<float>()),
    //     static_cast<int>(bbox_[1].item<float>()),
    //     static_cast<int>(bbox_[2].item<float>() - bbox_[0].item<float>()),
    //     static_cast<int>(bbox_[3].item<float>() - bbox_[1].item<float>())
    // );
    // cv::rectangle(img, bbox_rect, cv::Scalar(0, 255, 0), 2);
  }

  void set_destination(const BBox& dest_box) override {
    set_destination_size(dest_box.width(), dest_box.height());
  }

  constexpr const ResizingBoxConfig& resizing_config() const {
    return config_;
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

    constexpr float kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio = 6.0;
    constexpr float kMaxWidthHeightDiffDirectionCutRateRatio = 2.0;
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

  void adjust_size(float accel_w, float accel_h, bool use_constraints = true) {
    if (state_.size_is_frozen) {
      return;
    }

    if (use_constraints) {
      constexpr float kResizeLargerScaleDifference = 2.0;

      // Growing is allowed at a higher rate than shrinking
      const float max_accel_w = accel_w > 0
          ? (config_.max_accel_w * kResizeLargerScaleDifference)
          : config_.max_accel_w;
      const float max_accel_h = accel_h > 0
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

  const ResizingBoxConfig config_;
  ResizingState state_;
};

struct TranslatingBoxConfig {
  float max_speed_x{0.0};
  float max_speed_y{0.0};
  float max_accel_x{0.0};
  float max_accel_y{0.0};
  bool stop_on_dir_change{true};
  std::optional<BBox> arena_box{std::nullopt};
  std::optional<float> fixed_aspect_ratio{std::nullopt};
};

struct TranslationState {
  float current_speed_x{0.0};
  float current_speed_y{0.0};
  bool translation_is_frozen{false};
  // Nonstop stuff
  std::optional<int64_t> nonstop_delay{0};
  int64_t nonstop_delay_counter{0};
};

class TranslatingBox : public IBasicBox {
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
      state_.current_speed_x *= std::get<0>(x_y_on_edge);
      state_.current_speed_y *= std::get<1>(x_y_on_edge);
      total_diff.dx *= !!std::get<0>(x_y_on_edge);
      total_diff.dy *= !!std::get<1>(x_y_on_edge);
    }
  }

 private:
  void clamp_speed(float scale) {
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
      torch::optional<FloatValue> accel_x = c10::nullopt,
      torch::optional<FloatValue> accel_y = c10::nullopt,
      torch::optional<FloatValue> scale_constraints = c10::nullopt,
      torch::optional<FloatValue> nonstop_delay = c10::nullopt) {
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

  void scale_speed(
      torch::optional<FloatValue> ratio_x = c10::nullopt,
      torch::optional<FloatValue> ratio_y = c10::nullopt,
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

  float get_gaussian_y_about_width_center(float x) const {
    // Different than python
    if (!config_.arena_box.has_value()) {
      return 1.0;
    }
    x = clamp(x, gasussian_clamp_lr->first, gasussian_clamp_lr->second);
    const float center_x = config_.arena_box->width() / 2;
    if (x < center_x) {
      x = -(center_x - x);
    } else if (x > center_x) {
      x = x - center_x;
    } else {
      return 1.0;
    }
    return 1 - x / center_x;
  }

 private:
  TranslatingBoxConfig config_;
  TranslationState state_;
  std::optional<std::pair<float, float>> gasussian_clamp_lr{std::nullopt};
};

struct MovingBoxConfig {
  FloatValue scale_width{1.0};
  FloatValue scale_height{1.0};
};

struct AllMovingBoxConfig : public ResizingBoxConfig,
                            public TranslatingBoxConfig,
                            public MovingBoxConfig {};

class MovingBox : public BasicBox, public ResizingBox, public TranslatingBox {
 public:
  using BasicBox::bounding_box;
  using BasicBox::set_bbox;

  MovingBox(std::string label, BBox bbox, const AllMovingBoxConfig& config)
      : BasicBox(bbox),
        ResizingBox(config),
        TranslatingBox(config),
        label_(label),
        config_(config) {}

  WHDims get_size_scale() const {
    return WHDims{
        .width = config_.scale_width,
        .height = config_.scale_height,
    };
  }

  void draw(at::Tensor& img, bool draw_thresholds = false) override {
    // ResizingBox::draw(img, draw_thresholds);
    // TranslatingBox::draw(img, draw_thresholds);
    //  putText(img, label_, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
    //  cv::Scalar(255, 0, 0), 2);
  }

  void set_destination(const BBox& dest_box) override {
    ResizingBox::set_destination(dest_box);
    TranslatingBox::set_destination(dest_box);
    // set_destination_size(dest_box.width(), dest_box.height());
  }

  void set_destination(const IBasicBox& dest_box) {
    BBox bbox = dest_box.bounding_box();
    WHDims scale_box = get_size_scale();
    BBox scaled_bbox = bbox.make_scaled(scale_box.width, scale_box.height);
    ResizingBox::set_destination(scaled_bbox);
    TranslatingBox::set_destination(scaled_bbox);
  }

  // FloatValue get_zoom_level() {
  //   BBox bbox = bounding_box();
  //   FloatValue current_w = bbox.width();
  //   FloatValue current_h = bbox.height();
  //   return std::sqrt(current_w * current_w + current_h * current_h);
  // }

  BBox next_position() {
    // torch::Tensor dx = current_speed_x_;
    // torch::Tensor dy = current_speed_y_;

    // bbox_ += torch::tensor(std::vector<at::Tensor>{dx, dy, dx, dy},
    // torch::dtype(torch::kFloat).device(device_));

    clamp_to_arena();
    return bounding_box();
  }

  void clamp_to_arena() {
    const ResizingBoxConfig& rconfig = resizing_config();
    BBox bbox = bounding_box();
    auto z = zero();
    bbox.left = clamp(bbox.left, z, rconfig.max_width);
    bbox.top = clamp(bbox.top, z, rconfig.max_height);
    bbox.right = clamp(bbox.right, z, rconfig.max_width);
    bbox.bottom = clamp(bbox.bottom, z, rconfig.max_height);
    set_bbox(bbox);
  }

 private:
  const std::string label_;
  const MovingBoxConfig config_;
};

#if 0
int main() {
    // Example usage
    torch::Tensor bbox = torch::tensor({10, 10, 100, 100}, torch::kFloat);
    auto device = torch::kCPU;
    MovingBox moving_box("Example Box", bbox, 5.0, 5.0, 1.0, 1.0, 200.0, 200.0, true);

    at::Tensor img = at::Tensor::zeros(500, 500, CV_8UC3);
    moving_box.draw(img);

    cv::imshow("Moving Box", img);
    cv::waitKey(0);

    return 0;
}
#endif
