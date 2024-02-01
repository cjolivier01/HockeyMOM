#include "hockeymom/csrc/pytorch/image_blend.h"

#include <torch/nn/functional.h>

#include <iostream>

namespace hm {
namespace ops {

namespace {} // namespace

ImageBlender::ImageBlender(
    std::size_t levels,
    int x_pos_1,
    int y_pos_1,
    int x_pos_2,
    int y_pos_2,
    at::Tensor seam,
    at::Tensor xor_map,
    std::optional<std::string> interpolation)
    : levels_(levels),
      x_pos_1_(x_pos_1),
      y_pos_1_(x_pos_1),
      x_pos_2_(x_pos_2),
      y_pos_2_(y_pos_2),
      seam_(seam),
      xor_map_(xor_map),
      interpolation_(interpolation ? *interpolation : "") {}

void ImageBlender::init() {
  initialized_ = false;

  initialized_ = true;
}

void ImageBlender::to(std::string device) {
  assert(initialized_);
  seam_ = seam_.to(device);
  xor_map_ = xor_map_.to(device);
}

std::pair<at::Tensor, at::Tensor> ImageBlender::make_full(
    at::Tensor image_1,
    at::Tensor image_2) const {
  int h1 = image_1.size(2);
  int w1 = image_1.size(3);
  int x1 = x_pos_1_;
  int y1 = y_pos_1_;
  int h2 = image_2.size(2);
  int w2 = image_2.size(3);
  int x2 = x_pos_2_;
  int y2 = y_pos_2_;

  int canvas_w = seam_.size(1);
  int canvas_h = seam_.size(0);

  if (y1 < y2) {
    y2 -= y1;
    y1 = 0;
  } else if (y2 < y1) {
    y1 -= y2;
    y2 = 0;
  }
  if (x1 < x2) {
    x2 -= x1;
    x1 = 0;
  } else if (x2 < x1) {
    x1 -= x2;
    x2 = 0;
  }

  at::Tensor full_left = at::constant_pad_nd(
      image_1,
      {
          x1,
          canvas_w - x1 - w1,
          y1,
          canvas_h - y1 - h1,
      },
      0.0);

  at::Tensor full_right = at::constant_pad_nd(
      image_1,
      {
          x2,
          canvas_w - x2 - w2,
          y2,
          canvas_h - y2 - h2,
      },
      0.0);

  return {std::move(full_left), std::move(full_right)};
}

at::Tensor ImageBlender::hard_seam_blend(at::Tensor image_1, at::Tensor image_2)
    const {
  return image_1;
}

at::Tensor ImageBlender::forward(at::Tensor image_1, at::Tensor image_2) const {
  assert(initialized_);
  if (!levels_) {
    return hard_seam_blend(image_1, image_2);
  }
  return image_1.clone();
}

} // namespace ops
} // namespace hm
