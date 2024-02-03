#pragma once

#include <ATen/ATen.h>

#include <mutex>
#include <optional>
#include <string>

namespace hm {

namespace ops {

struct BlenderConfig {
  /**
   * @brief Modes: multiblend, hard_seam, laplacian
   */
  std::string mode = std::string("multiblend");
  int levels{0};
  at::Tensor seam;
  at::Tensor xor_map;
  std::string interpolation;
  std::string device = std::string("cpu");
};

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
  void init();
  void to(std::string device);
  bool is_initialized() const {
    return initialized_;
  }
  at::Tensor forward(
      at::Tensor&& image_1,
      std::vector<int> xy_pos_1,
      at::Tensor&& image_2,
      std::vector<int> xy_pos_2) const;

  std::pair<at::Tensor, at::Tensor> make_full(
      const at::Tensor& image_1,
      const std::vector<int>& xy_pos_1,
      const at::Tensor& image_2,
      const std::vector<int>& xy_pos_2) const;

 private:
  at::Tensor hard_seam_blend(
      at::Tensor&& image_1,
      std::vector<int> xy_pos_1,
      at::Tensor&& image_2,
      std::vector<int> xy_pos_2) const;

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
  std::string interpolation_;
};

} // namespace ops
} // namespace hm