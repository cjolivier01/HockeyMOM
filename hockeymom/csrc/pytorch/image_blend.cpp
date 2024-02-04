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

inline at::Tensor scalar_float(const float& val) {
  return torch::tensor({val});
}

at::Tensor create_gaussian_kernel(
    int kernel_size,
    int channels,
    float sigma = 1.0,
    at::ScalarType dtype = at::ScalarType::Float) {
  // Create Gaussian Kernel. In Numpy
  at::Tensor ax = at::linspace(
      -double(kernel_size - 1) / 2.0,
      double(kernel_size - 1) / 2.0,
      kernel_size);
  auto xx_and_yy = at::meshgrid({ax, ax});
  at::Tensor& xx = xx_and_yy.at(0);
  at::Tensor& yy = xx_and_yy.at(1);
  at::Tensor kernel_tensor = at::exp(
      -0.5 * (at::square(xx) + at::square(yy)) /
      at::square(scalar_float(sigma)));
  kernel_tensor = at::div(kernel_tensor, at::sum(kernel_tensor));
  // # Reshapes to (channels, 1, size, size)
  kernel_tensor = kernel_tensor.repeat({channels, 1, 1, 1});
  return kernel_tensor;
}

at::Tensor gaussian_conv2d(at::Tensor& x, torch::nn::Conv2d& conv) {
  TORCH_CHECK(
      x.dim() == 4,
      "Expected input tensor to be of shape: (batch, depth, height, width)");
  return conv->forward(x);
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
      interpolation_(interpolation ? *interpolation : ""),
      avg_pooling_(torch::nn::AvgPool2dOptions(/*kernel_size=*/2)) {
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

  if (mode_ == Mode::Laplacian) {
    gussian_kernel_ = create_gaussian_kernel(
        /*kernel_size=*/5,
        /*channels=*/3,
        /*sigma=*/1.0,
        /*dtype=*/at::ScalarType::Float);
    mask_gussian_kernel_ = create_gaussian_kernel(
        /*kernel_size=*/5,
        /*channels=*/1,
        /*sigma=*/1.0,
        /*dtype=*/at::ScalarType::Float);

    // Make the image conv op
    int channels = gussian_kernel_.size(0);
    int padding = gussian_kernel_.size(gussian_kernel_.dim() - 1) / 2;
    gaussian_conv_ = std::make_unique<torch::nn::Conv2d>(
        torch::nn::Conv2dOptions(channels, channels, gussian_kernel_.size(0))
            .stride(1)
            .padding(padding)
            .groups(channels));
    (*gaussian_conv_)->weight.set_data(gussian_kernel_);

    // Make the mask conv op
    channels = mask_gussian_kernel_.size(0);
    padding = mask_gussian_kernel_.size(mask_gussian_kernel_.dim() - 1) / 2;
    mask_gaussian_conv_ = std::make_unique<torch::nn::Conv2d>(
        torch::nn::Conv2dOptions(channels, channels, gussian_kernel_.size(0))
            .stride(1)
            .padding(padding)
            .groups(channels));
    (*mask_gaussian_conv_)->weight.set_data(mask_gussian_kernel_);
    create_masks();
  }

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
    if (mode_ == Mode::Laplacian) {
      (*gaussian_conv_)->to(device);
      (*mask_gaussian_conv_)->to(device);
      avg_pooling_->to(device);
    }
  }
}

at::Tensor ImageBlender::downsample(const at::Tensor& x) {
  return avg_pooling_->forward(x);
}

at::Tensor ImageBlender::upsample(at::Tensor& x, const SizeRef size) const {
  return torch::upsample_bilinear2d(x, size, /*align_corners=*/false);
}

std::vector<at::Tensor> ImageBlender::create_laplacian_pyramid(
    at::Tensor& x,
    torch::nn::Conv2d& conv) {
  std::vector<at::Tensor> pyramids;
  at::Tensor current_x = x;
  for (int level = 0; level < levels_; ++level) {
    std::cout << "current_x size: " << current_x.sizes() << std::endl;
    at::Tensor gauss_filtered_x = gaussian_conv2d(current_x, conv);
    std::cout << "gauss_filtered_x size: " << gauss_filtered_x.sizes()
              << std::endl;
    at::Tensor down = downsample(gauss_filtered_x);
    std::cout << "down size: " << down.sizes() << std::endl;
    at::Tensor laplacian = current_x -
        upsample(down,
                 {gauss_filtered_x.size(gauss_filtered_x.dim() - 2),
                  gauss_filtered_x.size(gauss_filtered_x.dim() - 1)});
    std::cout << "laplacian size: " << laplacian.sizes() << std::endl;
    pyramids.emplace_back(laplacian);
    current_x = down;
  }
  pyramids.emplace_back(current_x);
  return pyramids;
}

at::Tensor ImageBlender::one_level_gaussian_pyramid(
    at::Tensor& x,
    torch::nn::Conv2d& conv) {
  at::Tensor gauss_filtered_x = gaussian_conv2d(x, conv);
  return downsample(gauss_filtered_x);
}

void ImageBlender::create_masks() {
  at::Tensor mask = seam_.unsqueeze(0).unsqueeze(0);
  at::Tensor unique_values = std::get<0>(at::_unique(mask, /*sorted=*/true));
  TORCH_CHECK(
      unique_values.dim() == 1 and unique_values.size(0) == 2,
      "Need 2 unique values in the mask");
  at::Tensor left_value = unique_values[0];
  at::Tensor right_value = unique_values[1];
  mask.index_put_({mask == left_value}, 1.0);
  mask.index_put_({mask == right_value}, 0.0);
  mask = mask.to(at::ScalarType::Float);
  at::Tensor mask_img = mask;
  mask_small_gaussian_blurred_ = {mask.squeeze(0).squeeze(0)};
  for (int l = 0; l < levels_ + 1; ++l) {
    mask_img = one_level_gaussian_pyramid(mask_img, *mask_gaussian_conv_);
    mask_small_gaussian_blurred_.emplace_back(mask_img.squeeze(0).squeeze(0));
  }
  for (int i = 0; i < mask_small_gaussian_blurred_.size(); ++i) {
    mask_small_gaussian_blurred_[i] /= at::max(mask_small_gaussian_blurred_[i]);
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
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) const {
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
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) {
  assert(initialized_);
  if (mode_ == Mode::HardSeam) {
    return hard_seam_blend(
        std::move(image_1), xy_pos_1, std::move(image_2), xy_pos_2);
  }
  return laplacian_pyramid_blend(
      std::move(image_1), xy_pos_1, std::move(image_2), xy_pos_2);
}

at::Tensor ImageBlender::laplacian_pyramid_blend(
    at::Tensor&& image_1,
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) {
  auto [full_left, full_right] =
      make_full(image_1, xy_pos_1, image_2, xy_pos_2);

  full_left = full_left.to(at::ScalarType::Float);
  full_right = full_right.to(at::ScalarType::Float);

  std::cout << "full_left size=" << full_left.sizes()
            << "\nfull_right size=" << full_right.sizes() << std::endl;

  std::vector<at::Tensor> left_laplacian =
      create_laplacian_pyramid(full_left, *gaussian_conv_);
  std::vector<at::Tensor> right_laplacian =
      create_laplacian_pyramid(full_right, *gaussian_conv_);

  // std::cout << "left_laplacian size=" << left_laplacian.sizes()
  //           << "\nright_laplacian size=" << right_laplacian.sizes() <<
  //           std::endl;

  at::Tensor left_small_gaussian_blurred = *left_laplacian.rbegin();
  at::Tensor right_small_gaussian_blurred = *right_laplacian.rbegin();

  at::Tensor mask_1d = mask_small_gaussian_blurred_.at(levels_);
  at::Tensor mask_left = mask_1d;
  at::Tensor mask_right = 1 - mask_1d;

  at::Tensor F_2 = left_small_gaussian_blurred * mask_left +
      right_small_gaussian_blurred * mask_right;

  for (int this_level = levels_ - 1; this_level >= 0; this_level--) {
    at::Tensor mask_1d = mask_small_gaussian_blurred_.at(this_level);
    at::Tensor mask_left = mask_1d;
    at::Tensor mask_right = 1 - mask_1d;
    at::Tensor F_1 = upsample(
        F_2,
        {mask_1d.size(mask_1d.dim() - 2), mask_1d.size(mask_1d.dim() - 1)});
    at::Tensor upsampled_F1 = gaussian_conv2d(F_1, *gaussian_conv_);
    at::Tensor L_left = left_laplacian.at(this_level);
    at::Tensor L_right = right_laplacian.at(this_level);
    at::Tensor L_c = (mask_left * L_left) + (mask_right * L_right);
    F_2 = L_c + upsampled_F1;
  }

  return F_2;
}

} // namespace ops
} // namespace hm
