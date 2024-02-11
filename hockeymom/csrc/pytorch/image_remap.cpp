#include "hockeymom/csrc/pytorch/image_remap.h"

#include <torch/nn/functional.h>

#include <iostream>

namespace hm {
namespace ops {

namespace {

constexpr std::int64_t kUnmappedPixelValue = 65535;

template <typename VALUE_TYPE>
at::Tensor pad_tensor_to_size(
    at::Tensor& tensor,
    std::size_t target_width,
    std::size_t target_height,
    const VALUE_TYPE& pad_value) {
  std::size_t pad_width, pad_height;
  if (tensor.sizes().size() == 2) {
    pad_height = target_height - tensor.size(0);
    pad_width = target_width - tensor.size(1);
  } else {
    assert(tensor.sizes().size() == 3);
    pad_height = target_height - tensor.size(1);
    pad_width = target_width - tensor.size(2);
  }
  pad_height = std::max(0UL, pad_height);
  pad_width = std::max(0UL, pad_width);
  if (pad_width || pad_height) {
    return at::constant_pad_nd(
        tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
  }
  return tensor;
}

template <typename VALUE_TYPE>
at::Tensor pad_tensor_to_size_batched(
    at::Tensor& tensor,
    std::size_t target_width,
    std::size_t target_height,
    const VALUE_TYPE& pad_value) {
  std::size_t pad_width, pad_height;
  if (tensor.sizes().size() == 3) {
    pad_height = target_height - tensor.size(1);
    pad_width = target_width - tensor.size(2);
  } else {
    assert(tensor.sizes().size() == 4);
    pad_height = target_height - tensor.size(2);
    pad_width = target_width - tensor.size(3);
  }
  pad_height = std::max(0UL, pad_height);
  pad_width = std::max(0UL, pad_width);
  if (pad_width || pad_height) {
    return at::constant_pad_nd(
        tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
  }
  return tensor;
}
} // namespace

ImageRemapper::ImageRemapper(
    std::size_t src_width,
    std::size_t src_height,
    at::Tensor col_map,
    at::Tensor row_map,
    bool add_alpha_channel,
    std::optional<std::string> interpolation)
    : src_width_(src_width),
      src_height_(src_height),
      dest_width_(col_map.size(1)),
      dest_height_(col_map.size(0)),
      col_map_(col_map),
      row_map_(row_map),
      add_alpha_channel_(add_alpha_channel),
      interpolation_(interpolation ? *interpolation : "") {
  working_width_ = std::max(src_width_, dest_width_);
  working_height_ = std::max(src_height_, dest_height_);
}

void ImageRemapper::init(std::size_t batch_size) {
  if (initialized_) {
    return;
  }
  assert(!initialized_);
  initialized_ = false;
  // std::cout << "Padding tensors to size: " << working_width_ << " x "
  //           << working_height_ << std::endl;
  auto col_map = pad_tensor_to_size(
      col_map_, working_width_, working_height_, kUnmappedPixelValue);
  auto row_map = pad_tensor_to_size(
      row_map_, working_width_, working_height_, kUnmappedPixelValue);
  at::Tensor mask = at::logical_or(
      col_map == kUnmappedPixelValue, row_map == kUnmappedPixelValue);
  col_map.index_put_({mask}, 0);
  row_map.index_put_({mask}, 0);
  if (!interpolation_.empty()) {
    // Normalize to [-1, 1]
    at::Tensor row_map_normalized =
        (2.0 * row_map / ((float)working_height_ - 1.0)) - 1.0;
    at::Tensor col_map_normalized =
        (2.0 * col_map / ((float)working_width_ - 1.0)) - 1.0;
    // Create the grid for grid_sample
    auto grid = at::stack({col_map_normalized, row_map_normalized}, /*dim=*/-1);
    assert(grid.sizes().size() == 3);
    grid = grid.expand(
        {(int)batch_size, grid.size(0), grid.size(1), grid.size(2)});
    grid_ = grid.contiguous();
  }
  col_map_ = col_map.contiguous();
  row_map_ = row_map.contiguous();
  mask_ = mask.contiguous();
  if (add_alpha_channel_) {
    at::TensorOptions options;
    options = options.dtype(at::ScalarType::Byte);
    alpha_channel_ = at::empty(
        {(int)batch_size, 1, (int)working_height_, (int)working_width_},
        options);
    alpha_channel_.fill_(255);
    alpha_channel_.index_put_(
        {at::indexing::Slice(), at::indexing::Slice(), mask_}, (uint8_t)0);
  }
  initialized_ = true;
}

void ImageRemapper::to(at::Device device) {
  assert(initialized_);
  col_map_ = col_map_.to(device);
  row_map_ = row_map_.to(device);
  mask_ = mask_.to(device);
  if (grid_.defined()) {
    grid_ = grid_.to(device);
  }
  if (alpha_channel_.defined()) {
    alpha_channel_ = alpha_channel_.to(device);
  }
}

at::Tensor ImageRemapper::forward(at::Tensor source_tensor) const {
  assert(initialized_);
  source_tensor = pad_tensor_to_size_batched(
      source_tensor, working_width_, working_height_, (uint8_t)0);
  // batch + 3 channels
  assert(source_tensor.sizes().size() == 4);
  at::Tensor destination_tensor;
  if (interpolation_.empty()) {
    destination_tensor = at::empty_like(source_tensor);
    destination_tensor.index_put_(
        {at::indexing::Slice(), at::indexing::Slice()},
        source_tensor.index(
            {at::indexing::Slice(),
             at::indexing::Slice(),
             row_map_,
             col_map_}));
  } else {
    torch::nn::functional::GridSampleFuncOptions options;
    assert(interpolation_ == "bilinear"); // TODO: implement enum switch
    auto mode = torch::kBilinear;
    options =
        options.padding_mode(torch::kZeros).align_corners(false).mode(mode);
    destination_tensor = torch::nn::functional::grid_sample(
        source_tensor.to(at::TensorOptions().dtype(torch::kF32)),
        grid_,
        options);
    destination_tensor = destination_tensor.clamp(0, 255.0).to(torch::kByte);
  }
  destination_tensor.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), mask_}, (uint8_t)0);
  // Add alpha channel if necessary
  if (add_alpha_channel_) {
    destination_tensor =
        at::cat({destination_tensor, alpha_channel_}, /*dim=*/1);
  }
  // Clip to the original size that was specified
  destination_tensor = destination_tensor.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(0, dest_height_),
       torch::indexing::Slice(0, dest_width_)});
  return destination_tensor;
}

} // namespace ops
} // namespace hm
