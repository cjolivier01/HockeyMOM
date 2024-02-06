#pragma once

#include "hockeymom/csrc/mblend/threadpool.h"
#include "hockeymom/csrc/pytorch/image_blend.h"
#include "hockeymom/csrc/pytorch/image_remap.h"

#include <ATen/ATen.h>
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

struct RemapImageInfo {
  std::size_t src_width;
  std::size_t src_height;
  at::Tensor col_map;
  at::Tensor row_map;
  bool add_alpha_channel;
  cudaStream_t cuda_stream{nullptr};
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
      cudaStream_t result_stream,
      std::vector<RemapImageInfo> remap_image_info,
      ImageBlender::Mode blender_mode,
      std::size_t levels,
      at::Tensor seam,
      at::Tensor xor_map,
      bool lazy_init,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  at::Tensor forward(std::vector<StitchImageInfo> inputs);

 private:
  void init();
  absl::Mutex mu_;
  std::vector<RemapImageInfo> remap_image_infos_;
  std::vector<std::unique_ptr<ImageRemapper>> remappers_;
  std::unique_ptr<ImageBlender> blender_;
  std::unique_ptr<Eigen::ThreadPool> thread_pool_;
  std::unique_ptr<HmThreadPool> remap_thread_pool_;
};

} // namespace ops
} // namespace hm