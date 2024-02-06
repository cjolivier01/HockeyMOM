#pragma once

#include "hockeymom/csrc/mblend/threadpool.h"
#include "hockeymom/csrc/pytorch/image_blend.h"
#include "hockeymom/csrc/pytorch/image_remap.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cuda_runtime.h>

#include "absl/synchronization/mutex.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace hm {

namespace ops {

class StreamTensor {
 public:
  StreamTensor();
  StreamTensor(c10::cuda::CUDAStream stream, at::Tensor tensor);
  at::Tensor get();

 private:
  std::unique_ptr<c10::cuda::CUDAStream> stream_;
  at::Tensor tensor_;
};

struct RemapImageInfo {
  std::size_t src_width{0};
  std::size_t src_height{0};
  at::Tensor col_map;
  at::Tensor row_map;
  bool add_alpha_channel{false};
};

struct StitchImageInfo {
  at::Tensor image;
  std::vector<int> xy_pos;
};

class ImageStitcher {
  using SizeRef = at::IntArrayRef;

 public:
  // levels=0 = quick, hard seam
  ImageStitcher(
      std::vector<RemapImageInfo> remap_image_info,
      ImageBlender::Mode blender_mode,
      std::size_t levels,
      at::Tensor seam,
      at::Tensor xor_map,
      bool lazy_init,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  std::shared_ptr<StreamTensor> forward(std::vector<StitchImageInfo> inputs);

 private:
  std::vector<RemapImageInfo> remap_image_infos_;
  std::vector<std::unique_ptr<ImageRemapper>> remappers_;
  std::unique_ptr<ImageBlender> blender_;
  std::unique_ptr<Eigen::ThreadPool> thread_pool_;
  std::unique_ptr<HmThreadPool> remap_thread_pool_;
};

} // namespace ops
} // namespace hm