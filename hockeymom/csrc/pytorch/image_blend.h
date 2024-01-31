#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <string>

namespace hm {

// template <typename T>
// using Optional = std::optional<T>;

namespace ops {

struct BlendConfig {
  std::size_t src_width{0};
  std::size_t src_height{0};
  int x_pos_1{0};
  int y_pos_1{0};
  int x_pos_2{0};
  int y_pos_2{0};
  at::Tensor seam;
  at::Tensor xor_map;
  std::string interpolation;
  std::string device{"cpu"};
};

class ImageBlender {
 public:
  // levels=0 = quick, hard seam
  ImageBlender(
      std::size_t levels,
      int x_pos_1,
      int y_pos_1,
      int x_pos_2,
      int y_pos_2,
      at::Tensor seam,
      at::Tensor xor_map,
      std::optional<std::string> interpolation);
  void init();
  void to(std::string device);
  bool is_initialized() const {
    return initialized_;
  }
  at::Tensor forward(at::Tensor image_1, at::Tensor image_2) const;

 private:

  at::Tensor hard_seam_blend(at::Tensor image_1, at::Tensor image_2) const;
  std::pair<at::Tensor, at::Tensor> make_full(at::Tensor image_1, at::Tensor image_2) const;

  bool initialized_{false};
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