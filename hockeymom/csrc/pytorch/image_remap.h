#pragma once

#include <ATen/ATen.h>

namespace hm {
namespace ops {
at::Tensor add_tensors(const at::Tensor& a, const at::Tensor& b);
}
} // namespace hm