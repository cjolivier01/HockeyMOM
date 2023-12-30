#pragma once

#include <ATen/ATen.h>

namespace hm {
namespace ops {
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b);

class ImageRemapper {
 public:
  ImageRemapper(
      std::size_t src_width,
      std::size_t src_height,
      at::Tensor col_map,
      at::Tensor row_map,
      bool add_alpha_channel,
      std::string interpolation);
  void init(std::size_t batch_size);
  void to(std::string device);
  bool is_initialized() const {
    return initialized_;
  }
  at::Tensor remap(at::Tensor source_tensor);

 private:
  bool initialized_{false};
  std::size_t src_width_;
  std::size_t src_height_;
  std::size_t dest_width_;
  std::size_t dest_height_;
  std::size_t working_width_{0};
  std::size_t working_height_{0};
  at::Tensor col_map_;
  at::Tensor row_map_;
  at::Tensor mask_;
  at::Tensor grid_;
  at::Tensor alpha_channel_;
  bool add_alpha_channel_;
  std::string interpolation_;
};

} // namespace ops
} // namespace hm