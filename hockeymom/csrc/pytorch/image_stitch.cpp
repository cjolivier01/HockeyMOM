#include "hockeymom/csrc/pytorch/image_stitch.h"

#include <torch/torch.h>

namespace hm {
namespace ops {

namespace {

inline at::Tensor scalar_float(const float& val) {
  return torch::tensor({val});
}

constexpr std::size_t kRemapPoolThreadCount = 2;

} // namespace

ImageStitcher::ImageStitcher(
    cudaStream_t result_stream,
    std::vector<RemapImageInfo> remap_image_info,
    ImageBlender::Mode blender_mode,
    std::size_t levels,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : remap_image_infos_(std::move(remap_image_info)),
      thread_pool_(std::make_unique<Eigen::ThreadPool>(kRemapPoolThreadCount)),
      remap_thread_pool_(std::make_unique<HmThreadPool>(thread_pool_.get())) {
  for (const RemapImageInfo& remap_info : remap_image_infos_) {
    remappers_.emplace_back(std::make_unique<ImageRemapper>(
        remap_info.src_width,
        remap_info.src_height,
        remap_info.col_map,
        remap_info.row_map,
        remap_info.add_alpha_channel,
        interpolation));
  }
  blender_ = std::make_unique<ImageBlender>(
      blender_mode, levels, seam, xor_map, lazy_init, interpolation);
}

void ImageStitcher::init() {}

void ImageStitcher::to(at::Device device) {
  for (auto& m : remappers_) {
    m->to(device);
  }
  blender_->to(device);
}

at::Tensor ImageStitcher::forward(std::vector<StitchImageInfo> inputs) {
  // auto stream =  at::cuda::getStreamFromPool()
  for (std::size_t i = 0, n = inputs.size(); i < n; ++i) {
  }

  return at::Tensor();
}

} // namespace ops
} // namespace hm
