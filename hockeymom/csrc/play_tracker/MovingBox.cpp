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

struct BBox {
  float left{0.0};
  float top{0.0};
  float right{0.0};
  float bottom{0.0};
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
};

// using BBox = torch::Tensor;
// using FloatValue = torch::Tensor;
using FloatValue = float;

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
  const float size_ratio_thresh_grow_dw_{0.05};
  // Threshold to grow height (ratio of bbox)
  const float size_ratio_thresh_grow_dy_{0.1};
  // Threshold to shrink width (ratio of bbox)
  const float size_ratio_thresh_shrink_dw_{0.08};
  // Threshold to shrink height (ratio of bbox)
  const float size_ratio_thresh_shrink_dy_{0.1};
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
  void set_destination_size(FloatValue dest_width, FloatValue dest_height) {
    BBox bbox = bounding_box();
    auto current_w = bbox.width();
    auto current_h = bbox.height();

    auto dw = dest_width - current_w;
    auto dh = dest_height - current_h;

    adjust_size(dw, dh);
  }

  void adjust_size(
      std::optional<float> accel_w,
      std::optional<float> accel_h,
      bool use_constraints = true) {
    if (accel_w.has_value()) {
      state_.current_speed_w += *accel_w;
    }

    if (accel_h.has_value()) {
      state_.current_speed_h += *accel_h;
    }

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

  const ResizingBoxConfig config_;
  ResizingState state_;
};

struct TranslatingBoxConfig {
  float max_speed_x{0.0};
  float max_speed_y{0.0};
  float max_accel_x{0.0};
  float max_accel_y{0.0};
  bool stop_on_dir_change{true};
};

struct TranslationState {
  float current_speed_x{0.0};
  float current_speed_y{0.0};
};

struct MovingBoxConfig : public ResizingBoxConfig,
                         public TranslatingBoxConfig {};

class TranslatingBox : public IBasicBox {
 public:
  TranslatingBox(const TranslatingBoxConfig& config) : config_(config) {}

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
    if (accel_x.has_value()) {
      state_.current_speed_x += *accel_x;
    }

    if (accel_y.has_value()) {
      state_.current_speed_y += *accel_y;
    }

    if (scale_constraints.has_value()) {
      clamp_speed(*scale_constraints);
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

 private:
  TranslatingBoxConfig config_;
  TranslationState state_;
};

class MovingBox : public BasicBox, public ResizingBox, public TranslatingBox {
 public:
  using BasicBox::bounding_box;
  using BasicBox::set_bbox;

  MovingBox(std::string label, BBox bbox, const MovingBoxConfig& config)
      : BasicBox(bbox),
        ResizingBox(config),
        TranslatingBox(config),
        label_(label) {}

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

  FloatValue get_zoom_level() {
    BBox bbox = bounding_box();
    FloatValue current_w = bbox.width();
    FloatValue current_h = bbox.height();
    return std::sqrt(current_w * current_w + current_h * current_h);
  }

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
