#include "hockeymom/csrc/pytorch/image_blend.h"

#include <torch/nn/functional.h>

#include <iostream>

namespace hm {
namespace ops {

namespace {} // namespace

ImageBlender::ImageBlender(
    Mode mode,
    std::size_t levels,
    at::Tensor seam,
    at::Tensor xor_map,
    std::optional<std::string> interpolation)
    : mode_(mode),
      levels_(levels),
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
    const at::Tensor& image_1,
    const std::vector<int>& xy_pos_1,
    const at::Tensor& image_2,
    const std::vector<int>& xy_pos_2) const {
  assert(image_1.dim() == 3);
  assert(image_1.size(2) == 3 || image_1.size(2) == 4);
  int h1 = image_1.size-(0);
  int w1 = image_1.size(1);
  int x1 = xy_pos_1.at(0);
  int y1 = xy_pos_1.at(1);
  int h2 = image_2.size(0);
  int w2 = image_2.size(1);
  int x2 = xy_pos_2.at(0);
  int y2 = xy_pos_2.at(1);

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

at::Tensor ImageBlender::hard_seam_blend(
    at::Tensor&& image_1,
    std::vector<int> xy_pos_1,
    at::Tensor&& image_2,
    std::vector<int> xy_pos_2) const {
  auto [full_left, full_right] =
      make_full(image_1, xy_pos_1, image_2, xy_pos_2);

  int channels = image_1.size(1);
  assert(channels == 3 || channels == 4);
  at::TensorOptions options;
  options = options.dtype(image_1.dtype()).device(image_1.device());
  at::Tensor canvas = at::empty(
      {image_1.size(0), // batch size
       channels,
       seam_.size(0),
       seam_.size(1)},
      options);

  // canvas[:, :, self._seam_mask == self._left_value] = full_left[
  //     :, :, self._seam_mask == self._left_value
  // ]
  // canvas[:, :, self._seam_mask == self._right_value] = full_right[
  //     :, :, self._seam_mask == self._right_value
  // ]
  at::Tensor condition_left = seam_ == left_seam_value_;
  canvas.index_put_({condition_left}, full_left.index({condition_left}));
  at::Tensor condition_right = seam_ == right_seam_value_;
  canvas.index_put_({condition_right}, full_left.index({condition_right}));

  return canvas;
}

at::Tensor ImageBlender::forward(
    at::Tensor&& image_1,
    std::vector<int> xy_pos_1,
    at::Tensor&& image_2,
    std::vector<int> xy_pos_2) const {
  assert(initialized_);
  if (!levels_) {
    return hard_seam_blend(
        std::move(image_1),
        std::move(xy_pos_1),
        std::move(image_2),
        std::move(xy_pos_2));
  }
  assert(false);
  return image_1.clone();
}

} // namespace ops
} // namespace hm
