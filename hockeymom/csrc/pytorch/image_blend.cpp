#include "hockeymom/csrc/pytorch/image_blend.h"

#include <torch/nn/functional.h>

//#include <iostream>

namespace hm {
namespace ops {

namespace {

int constrain_index(int max_index, int calculated_index) {
  return calculated_index;
  // int final_index = calculated_index;
  // if (final_index < 0) {
  //   final_index = max_index + final_index;
  // }
  // TORCH_CHECK(final_index <= max_index, "Calculated index is too large");
  // return final_index;
}

} // namespace

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
      interpolation_(interpolation ? *interpolation : "") {
  init();
}

void ImageBlender::init() {
  initialized_ = false;
  TORCH_CHECK(
      seam_.ndimension() == 2,
      "Seam tensor should be two dimensions only (h, w)");
  auto unique_results = at::_unique(seam_, /*sorted=*/true);
  torch::Tensor unique_elements = std::get<0>(unique_results);
  assert(unique_elements.size(0) == 2);
  assert(unique_elements.dim() == 1);
  left_seam_value_ = unique_elements[0];
  condition_left_ = at::eq(seam_, left_seam_value_);
  right_seam_value_ = unique_elements[1];
  condition_right_ = at::eq(seam_, right_seam_value_);
  initialized_ = true;
}

void ImageBlender::to(at::Device device) {
  seam_ = seam_.to(device);
  xor_map_ = xor_map_.to(device);
  if (initialized_) {
    left_seam_value_ = left_seam_value_.to(device);
    condition_left_ = condition_left_.to(device);
    right_seam_value_ = right_seam_value_.to(device);
    condition_right_ = condition_right_.to(device);
  }
}

std::pair<at::Tensor, at::Tensor> ImageBlender::make_full(
    const at::Tensor& image_1,
    const std::vector<int>& xy_pos_1,
    const at::Tensor& image_2,
    const std::vector<int>& xy_pos_2) const {
  assert(image_1.dim() == 4);
  assert(image_1.size(1) == 3 || image_1.size(0) == 4);
  int h1 = image_1.size(2);
  int w1 = image_1.size(3);
  int x1 = xy_pos_1.at(0);
  int y1 = xy_pos_1.at(1);
  int h2 = image_2.size(2);
  int w2 = image_2.size(3);
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

  // std::cout << "Canvas size=[" << canvas_h << ", " << canvas_w << "]"
  //           << std::endl;
  // std::cout << "image_1 size=" << image_1.sizes()
  //           << "\nimage_2 size=" << image_2.sizes() << std::endl;

  TORCH_CHECK(x1 == 0 || x2 == 0, "Images not aligned to left edge of canvas");
  TORCH_CHECK(y1 == 0 || y2 == 0, "Images not aligned to top edge of canvas");
  TORCH_CHECK(x1 + w1 <= canvas_w, "First image overflows the canvas width");
  TORCH_CHECK(y1 + h1 <= canvas_h, "First image overflows the canvas height");
  TORCH_CHECK(x2 + w2 <= canvas_w, "Second image overflows the canvas width");
  TORCH_CHECK(y2 + h2 <= canvas_h, "Second image overflows the canvas height");

  TORCH_CHECK(x1 <= w1, "Invalid x1: " + std::to_string(x1));
  TORCH_CHECK(x1 <= h1, "Invalid y1: " + std::to_string(y1));
  TORCH_CHECK(x2 <= w2, "Invalid x2:" + std::to_string(x2));
  TORCH_CHECK(y2 <= h2, "Invalid y2: " + std::to_string(y2));

  at::Tensor full_left = at::constant_pad_nd(
      image_1,
      {
          x1,
          constrain_index(canvas_w, canvas_w - (w1 - x1)),
          y1,
          constrain_index(canvas_h, canvas_h - (h1 - y1)),
      },
      0.0);

  at::Tensor full_right = at::constant_pad_nd(
      image_2,
      {
          x2,
          constrain_index(canvas_w, canvas_w - (w2 + x2)),
          y2,
          constrain_index(canvas_h, canvas_h - (h2 + y2)),
      },
      0.0);

  // std::cout << "full_left size=" << full_left.sizes()
  //           << "\nfull_right size=" << full_right.sizes() << std::endl;

  TORCH_CHECK(full_left.size(2) == canvas_h);
  TORCH_CHECK(full_left.size(3) == canvas_w);
  TORCH_CHECK(
      full_left.sizes() == full_right.sizes(),
      "Full left and right sizes must be the same");

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

  // std::cout << "seam size=" << seam_.sizes() << std::endl;
  // std::cout << "canvas size=" << canvas.sizes() << std::endl;

  // std::cout << seam_.sizes() << ", " << seam_.dtype() << std::endl;
  // std::cout << seam_.sizes() << ", " << seam_.dtype() << std::endl;
  // std::cout << left_seam_value_.sizes() << ", " << left_seam_value_.dtype()
  //           << std::endl;

  canvas.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), condition_left_},
      full_left.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           condition_left_}));
  canvas.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), condition_right_},
      full_right.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           condition_right_}));

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
