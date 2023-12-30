#include "hockeymom/csrc/pytorch/image_remap.h"

namespace hm {
namespace ops {
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b) {
  return a + b;
}

ImageRemapper::ImageRemapper(
    at::Tensor col_map,
    at::Tensor row_map,
    at::Device device,
    bool add_alpha_channel,
    std::optional<std::string> interpolation)
    : col_map_(col_map),
      row_map_(row_map),
      device_(device),
      add_alpha_channel_(add_alpha_channel),
      interpolation_(interpolation ? *interpolation : "") {}

at::Tensor ImageRemapper::remap(at::Tensor source_tensor) {
  return source_tensor.clone();
}

} // namespace ops
} // namespace hm
