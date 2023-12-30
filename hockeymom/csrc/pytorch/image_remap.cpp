#include "hockeymom/csrc/pytorch/image_remap.h"

// #define TORCH_EXTENSION_NAME "hockeymom_ext"

namespace hm {
namespace ops {
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b) {
  return a + b;
}
} // namespace ops
} // namespace hm

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("add", &add_tensors, "A function that adds two tensors");
// }
