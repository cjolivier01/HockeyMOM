#pragma once

#include <ATen/ATen.h>

namespace hm {
namespace ops {
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b);

class ImageRemapper {
 public:
  ImageRemapper(
      at::Tensor col_map,
      at::Tensor row_map,
      at::Device device,
      bool add_alpha_channel,
      std::optional<std::string> interpolation);
  void init();
  bool is_initialized() const { return initialized_; }
  at::Tensor remap(at::Tensor source_tensor);
 private:
  bool initialized_{false};
  at::Tensor col_map_;
  at::Tensor row_map_;
  at::Device device_;
  bool add_alpha_channel_;
  std::string interpolation_;
};

} // namespace ops
} // namespace hm