#include "hockeymom/csrc/pytorch/image_blend.h"

#include <torch/nn/functional.h>

#include <iostream>

namespace hm {
namespace ops {

namespace {

// constexpr std::int64_t kUnmappedPixelValue = 65535;

// template <typename VALUE_TYPE>
// at::Tensor pad_tensor_to_size(
//     at::Tensor& tensor,
//     std::size_t target_width,
//     std::size_t target_height,
//     const VALUE_TYPE& pad_value) {
//   std::size_t pad_width, pad_height;
//   if (tensor.sizes().size() == 2) {
//     pad_height = target_height - tensor.size(0);
//     pad_width = target_width - tensor.size(1);
//   } else {
//     assert(tensor.sizes().size() == 3);
//     pad_height = target_height - tensor.size(1);
//     pad_width = target_width - tensor.size(2);
//   }
//   pad_height = std::max(0UL, pad_height);
//   pad_width = std::max(0UL, pad_width);
//   if (pad_width || pad_height) {
//     return at::constant_pad_nd(
//         tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
//   }
//   return tensor;
// }

// template <typename VALUE_TYPE>
// at::Tensor pad_tensor_to_size_batched(
//     at::Tensor& tensor,
//     std::size_t target_width,
//     std::size_t target_height,
//     const VALUE_TYPE& pad_value) {
//   std::size_t pad_width, pad_height;
//   if (tensor.sizes().size() == 3) {
//     pad_height = target_height - tensor.size(1);
//     pad_width = target_width - tensor.size(2);
//   } else {
//     assert(tensor.sizes().size() == 4);
//     pad_height = target_height - tensor.size(2);
//     pad_width = target_width - tensor.size(3);
//   }
//   pad_height = std::max(0UL, pad_height);
//   pad_width = std::max(0UL, pad_width);
//   if (pad_width || pad_height) {
//     return at::constant_pad_nd(
//         tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
//   }
//   return tensor;
// }
} // namespace

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
  //int h1 = image_1.shape[2];
  return {};
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
