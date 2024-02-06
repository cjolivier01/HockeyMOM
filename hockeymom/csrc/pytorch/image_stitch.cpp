#include "hockeymom/csrc/pytorch/image_stitch.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

namespace hm {
namespace ops {

namespace {

inline at::Tensor scalar_float(const float& val) {
  return torch::tensor({val});
}

} // namespace

StreamTensor::StreamTensor(c10::cuda::CUDAStream stream, at::Tensor tensor)
    : stream_(std::make_unique<c10::cuda::CUDAStream>(std::move(stream))),
      tensor_(std::move(tensor)) {}

at::Tensor StreamTensor::get() {
  if (stream_) {
    stream_->synchronize();
    stream_.reset();
  }
  return tensor_;
}

struct HmCudaStreamGuard {
 HmCudaStreamGuard(cudaStream_t stream) : stream_(stream) {

 }
 ~HmCudaStreamGuard() {

 }
 private:
  cudaStream_t stream_;
};

ImageStitcher::ImageStitcher(
    std::size_t batch_size,
    std::vector<RemapImageInfo> remap_image_info,
    ImageBlender::Mode blender_mode,
    std::size_t levels,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : remap_image_infos_(std::move(remap_image_info)),
      thread_pool_(
          std::make_unique<Eigen::ThreadPool>(remap_image_infos_.size())),
      remap_thread_pool_(std::make_unique<HmThreadPool>(thread_pool_.get())) {
  for (const RemapImageInfo& remap_info : remap_image_infos_) {
    remappers_.emplace_back(std::make_unique<ImageRemapper>(
        remap_info.src_width,
        remap_info.src_height,
        remap_info.col_map,
        remap_info.row_map,
        remap_info.add_alpha_channel,
        interpolation));
    (*remappers_.rbegin())->init(batch_size);
  }
  blender_ = std::make_unique<ImageBlender>(
      blender_mode, levels, seam, xor_map, lazy_init, interpolation);
}

void ImageStitcher::to(at::Device device) {
  for (auto& m : remappers_) {
    m->to(device);
  }
  blender_->to(device);
}

at::Tensor ImageStitcher::forward(
    std::vector<StitchImageInfo> inputs) {
  // if (!initialized_) {
  //   int batch_size = inputs.at(0).image.size(0);
  //   for (auto& r : remappers_) {
  //     r->init(batch_size);
  //   }
  // }
  HmThreadPool thread_pool(*remap_thread_pool_);
  std::vector<StreamTensor> remap_tensors(inputs.size());
  for (std::size_t i = 0, n = inputs.size(); i < n; ++i) {
    thread_pool.Schedule([this, i, &remap_tensors, &inputs]() {
      StitchImageInfo& img_info = inputs.at(i);
      c10::cuda::CUDAStream remap_stream = img_info.cuda_stream.has_value()
          ? std::move(img_info.cuda_stream.value())
          : at::cuda::getStreamFromPool();
      // Set the current stream
      c10::cuda::CUDAStreamGuard stream_guard(remap_stream);
      at::Tensor remapped_tensor =
          remappers_.at(i)->forward(inputs.at(i).image);
      remap_tensors.at(i) = StreamTensor(remap_stream, remapped_tensor);
    });
  }
  thread_pool.join_all();
  // c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
  // c10::cuda::CUDAStreamGuard stream_guard(stream);
  assert(remap_tensors.size() == 2);
  at::Tensor stitched_tensor = blender_->forward(
      std::move(remap_tensors.at(0).get()),
      inputs.at(0).xy_pos,
      std::move(remap_tensors.at(0).get()),
      inputs.at(1).xy_pos);

  //return std::make_shared<StreamTensor>(stream, stitched_tensor);
  return stitched_tensor;
}

} // namespace ops
} // namespace hm
