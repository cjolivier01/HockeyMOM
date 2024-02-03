#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <optional>
#include <string>

namespace hm {

namespace ops {

class ImageBlender {
 public:
  enum class Mode { HardSeam, Laplacian };

  // levels=0 = quick, hard seam
  ImageBlender(
      Mode mode,
      std::size_t levels,
      at::Tensor seam,
      at::Tensor xor_map,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  at::Tensor forward(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2) const;

  std::pair<at::Tensor, at::Tensor> make_full(
      const at::Tensor& image_1,
      const std::vector<int>& xy_pos_1,
      const at::Tensor& image_2,
      const std::vector<int>& xy_pos_2) const;

 private:
  void init();
  at::Tensor hard_seam_blend(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2) const;

  at::Tensor laplacian_pyramid_blend(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2) const;

  bool initialized_{false};
  Mode mode_;
  std::size_t levels_;
  std::size_t src_width_;
  std::size_t src_height_;

  at::Tensor seam_;
  at::Tensor xor_map_;
  at::Tensor left_seam_value_;
  at::Tensor condition_left_;
  at::Tensor right_seam_value_;
  at::Tensor condition_right_;
  std::vector<at::Tensor> seam_masks_;

  // Laplacian pyramid persistent tensors
  at::Tensor gussian_kernel;
  at::Tensor mask_gussian_kernel;
  //torch::nn::Conv2d gaussian_conv_;

  std::string interpolation_;
};

} // namespace ops
} // namespace hm