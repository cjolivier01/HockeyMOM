#include "hockeymom/csrc/pytorch/image_stitch.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

namespace hm {
namespace ops {

StreamTensor::StreamTensor(c10::cuda::CUDAStream stream, at::Tensor tensor)
    : stream_(std::make_unique<c10::cuda::CUDAStream>(std::move(stream))),
      tensor_(std::move(tensor)) {}

StreamTensor::StreamTensor(at::Tensor tensor) : tensor_(std::move(tensor)) {}

at::Tensor StreamTensor::get() {
  if (stream_) {
    stream_->synchronize();
    stream_.reset();
  }
  return tensor_;
}

struct HmCudaStreamGuard {
  HmCudaStreamGuard(cudaStream_t stream) : stream_(stream) {}
  ~HmCudaStreamGuard() {}

 private:
  cudaStream_t stream_;
};

ImageStitcher::ImageStitcher(
    std::size_t batch_size,
    std::vector<RemapImageInfo> remap_image_info,
    ImageBlender::Mode blender_mode,
    bool half,
    std::size_t levels,
    bool remap_on_async_stream,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : dtype_(half ? at::ScalarType::Half : at::ScalarType::Float),
      remap_image_infos_(std::move(remap_image_info)),
      thread_pool_(
          std::make_unique<Eigen::ThreadPool>(remap_image_infos_.size())),
      remap_thread_pool_(std::make_unique<HmThreadPool>(thread_pool_.get())),
      remap_on_async_stream_(remap_on_async_stream) {
  for (const RemapImageInfo& remap_info : remap_image_infos_) {
    remappers_.emplace_back(std::make_unique<ImageRemapper>(
        remap_info.src_width,
        remap_info.src_height,
        remap_info.col_map,
        remap_info.row_map,
        dtype_,
        remap_info.add_alpha_channel,
        remap_info.pad_value,
        interpolation));
    (*remappers_.rbegin())->init(batch_size);
  }
  blender_ = std::make_unique<ImageBlender>(
      blender_mode,
      half,
      levels,
      seam,
      xor_map,
      /*lazy_init=*/true,
      interpolation);
}

void ImageStitcher::to(at::Device device) {
  for (auto& m : remappers_) {
    m->to(device);
  }
  blender_->to(device);
}

at::Tensor ImageStitcher::forward(std::vector<StitchImageInfo> inputs) {
  if (!initialized_) {
    int batch_size = inputs.at(0).image.size(0);
    for (auto& r : remappers_) {
      r->init(batch_size);
    }
  }
  for (auto& t : inputs) {
    TORCH_CHECK(
        torch::is_floating_point(t.image), "Inputs must be floating point");
  }
  std::vector<StreamTensor> remap_tensors(inputs.size());
  if (remap_on_async_stream_) {
    at::cuda::CUDAStream current_stream =
        at::cuda::getCurrentCUDAStream(inputs.at(0).image.device().index());
    current_stream.synchronize();
    HmThreadPool thread_pool(*remap_thread_pool_);
    for (std::size_t i = 0, n = inputs.size(); i < n; ++i) {
      thread_pool.Schedule([this, i, &remap_tensors, &inputs]() {
        StitchImageInfo& img_info = inputs.at(i);
        c10::cuda::CUDAStream remap_stream = at::cuda::getStreamFromPool();
        // Set the current stream
        c10::cuda::CUDAStreamGuard stream_guard(remap_stream);
        at::Tensor remapped_tensor =
            remappers_.at(i)->forward(inputs.at(i).image);
        remap_tensors.at(i) = StreamTensor(remap_stream, remapped_tensor);
      });
    }
    thread_pool.join_all();
  } else {
    for (std::size_t i = 0, n = remappers_.size(); i < n; ++i) {
      at::Tensor remapped_tensor =
          remappers_.at(i)->forward(inputs.at(i).image);
      remap_tensors.at(i) = StreamTensor(remapped_tensor);
    }
  }
  // return remap_tensors.at(0).get();
  assert(remap_tensors.size() == 2);
  at::Tensor stitched_tensor = blender_->forward(
      std::move(remap_tensors.at(0).get()),
      inputs.at(0).xy_pos,
      std::move(remap_tensors.at(1).get()),
      inputs.at(1).xy_pos);

  return stitched_tensor;
}

} // namespace ops
} // namespace hm
