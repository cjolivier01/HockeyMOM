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
  int x_pos{0};
  int y_pos{0};
  at::Tensor seam;
  at::Tensor xor_map;
  std::string interpolation;
  std::size_t batch_size{1};
  std::string device{"cpu"};
};

class ImageBlender {
 public:
  ImageBlender(
      std::size_t src_width,
      std::size_t src_height,
      at::Tensor col_map,
      at::Tensor row_map,
      std::optional<std::string> interpolation);
  void init(std::size_t batch_size);
  void to(std::string device);
  bool is_initialized() const {
    return initialized_;
  }
  at::Tensor forward(at::Tensor image_1, at::Tensor image_2) const;

 private:
  bool initialized_{false};
  std::size_t src_width_;
  std::size_t src_height_;
  std::size_t dest_width_;
  std::size_t dest_height_;
  at::Tensor seam_;
  at::Tensor xor_map_;
  std::string interpolation_;
};

} // namespace ops
} // namespace hm