#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <string>

namespace hm {

// template <typename T>
// using Optional = std::optional<T>;

namespace ops {

struct BlenderConfig {
  /**
   * @brief Modes: multiblend, hard_seam, laplacian
   */
  std::string mode{"multiblend"};
  int levels{0};
  at::Tensor seam;
  at::Tensor xor_map;
  std::string interpolation;
  std::string device{"cpu"};
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

 private:
  at::Tensor hard_seam_blend(
      at::Tensor&& image_1,
      std::vector<int> xy_pos_1,
      at::Tensor&& image_2,
      std::vector<int> xy_pos_2) const;

  std::pair<at::Tensor, at::Tensor> make_full(
      const at::Tensor& image_1,
      const std::vector<int>& xy_pos_1,
      const at::Tensor& image_2,
      const std::vector<int>& xy_pos_2) const;

  bool initialized_{false};
  Mode mode_;
  std::size_t levels_;
  std::size_t src_width_;
  std::size_t src_height_;

  at::Tensor seam_;
  at::Tensor xor_map_;
  at::Tensor left_seam_value_;
  at::Tensor right_seam_value_;
  std::vector<at::Tensor> seam_masks_;
  std::string interpolation_;
};

} // namespace ops
} // namespace hm