#include "BYTETrackerCudaStatic.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace hm {
namespace tracker {
namespace {

constexpr const char* kFrameIdKey = "frame_id";
constexpr const char* kBBoxesKey = "bboxes";
constexpr const char* kLabelsKey = "labels";
constexpr const char* kScoresKey = "scores";
constexpr const char* kIdsKey = "ids";
constexpr const char* kNumDetectionsKey = "num_detections";
constexpr const char* kNumTracksKey = "num_tracks";

int64_t tensor_length(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.numel() >= 1,
      name,
      " tensor must contain at least one value");
  return tensor.to(at::kLong).item<int64_t>();
}

at::Tensor slice_rows(
    const at::Tensor& tensor,
    int64_t expected_rows,
    int64_t valid_rows) {
  TORCH_CHECK(
      tensor.dim() >= 1,
      "tensor must have at least one dimension");
  TORCH_CHECK(
      tensor.size(0) == expected_rows,
      "Expected tensor with first dimension ",
      expected_rows,
      ", got ",
      tensor.size(0));
  if (valid_rows == 0) {
    return tensor.slice(0, 0, 0);
  }
  return tensor.slice(0, 0, valid_rows);
}

template <typename T>
void copy_prefix(at::Tensor& dst, const at::Tensor& src, T count) {
  if (count <= 0) {
    return;
  }
  dst.slice(0, 0, count).copy_(src);
}

at::Tensor build_index_tensor_cuda(const at::Tensor& mask) {
  auto mask_bool = mask.to(at::kBool).contiguous();
  auto numel = mask_bool.numel();
  auto options = at::TensorOptions().dtype(at::kLong).device(mask.device());
  auto out = at::empty({numel}, options);
  if (numel == 0) {
    return out.narrow(0, 0, 0);
  }
  auto stream = at::cuda::getCurrentCUDAStream(mask.device().index());
  auto begin = thrust::make_counting_iterator<int64_t>(0);
  auto mask_ptr = mask_bool.data_ptr<bool>();
  auto out_ptr = out.data_ptr<int64_t>();
  thrust::device_ptr<const bool> mask_dev(mask_ptr);
  thrust::device_ptr<int64_t> out_dev(out_ptr);
  auto end = thrust::copy_if(
      thrust::cuda::par.on(stream.stream()),
      begin,
      begin + numel,
      mask_dev,
      out_dev,
      [] __device__(bool keep) { return keep; });
  auto count = static_cast<int64_t>(end - out_dev);
  return out.narrow(0, 0, count);
}

at::Tensor build_index_tensor_cpu(const at::Tensor& mask) {
  auto mask_bool = mask.to(at::kBool).contiguous();
  auto numel = mask_bool.numel();
  auto options = at::TensorOptions().dtype(at::kLong).device(mask.device());
  auto out = at::empty({numel}, options);
  auto mask_ptr = mask_bool.data_ptr<bool>();
  auto out_ptr = out.data_ptr<int64_t>();
  int64_t count = 0;
  for (int64_t i = 0; i < numel; ++i) {
    if (mask_ptr[i]) {
      out_ptr[count++] = i;
    }
  }
  return out.narrow(0, 0, count);
}

} // namespace

BYTETrackerCudaStatic::BYTETrackerCudaStatic(
    ByteTrackConfig config,
    int64_t max_detections,
    int64_t max_tracks,
    c10::Device device)
    : BYTETrackerCuda(std::move(config), device),
      max_detections_(max_detections),
      max_tracks_(max_tracks) {
  TORCH_CHECK(max_detections_ > 0, "max_detections must be positive");
  TORCH_CHECK(max_tracks_ > 0, "max_tracks must be positive");
}

std::unordered_map<std::string, at::Tensor> BYTETrackerCudaStatic::track(
    std::unordered_map<std::string, at::Tensor>&& data) {
  auto frame_id_tensor = data.at(kFrameIdKey);
  auto num_det_it = data.find(kNumDetectionsKey);
  TORCH_CHECK(
      num_det_it != data.end(),
      "data must contain 'num_detections' entry");
  int64_t num_detections = tensor_length(num_det_it->second, "num_detections");
  TORCH_CHECK(
      num_detections >= 0 && num_detections <= max_detections_,
      "num_detections (",
      num_detections,
      ") exceeds configured max_detections (",
      max_detections_,
      ")");

  auto det_bboxes = data.at(kBBoxesKey);
  auto det_labels = data.at(kLabelsKey);
  auto det_scores = data.at(kScoresKey);

  TORCH_CHECK(
      det_bboxes.dim() == 2 && det_bboxes.size(1) == 4,
      "bboxes tensor must have shape [max_detections, 4]");
  TORCH_CHECK(
      det_labels.dim() == 1,
      "labels tensor must be 1-D");
  TORCH_CHECK(
      det_scores.dim() == 1,
      "scores tensor must be 1-D");
  TORCH_CHECK(
      det_labels.size(0) == max_detections_,
      "labels tensor first dimension must equal max_detections");
  TORCH_CHECK(
      det_scores.size(0) == max_detections_,
      "scores tensor first dimension must equal max_detections");

  std::unordered_map<std::string, at::Tensor> tracker_input;
  tracker_input.emplace(kFrameIdKey, frame_id_tensor);
  auto trimmed_bboxes = slice_rows(det_bboxes, max_detections_, num_detections)
                            .to(device(), at::kFloat);
  auto trimmed_labels = slice_rows(det_labels, max_detections_, num_detections)
                            .to(device(), at::kLong);
  auto trimmed_scores = slice_rows(det_scores, max_detections_, num_detections)
                            .to(device(), at::kFloat);

  tracker_input.emplace(kBBoxesKey, trimmed_bboxes);
  tracker_input.emplace(kLabelsKey, trimmed_labels);
  tracker_input.emplace(kScoresKey, trimmed_scores);

  auto tracker_output = run_tracker(std::move(tracker_input));

  auto ids = tracker_output.at(kIdsKey);
  auto labels = tracker_output.at(kLabelsKey);
  auto scores = tracker_output.at(kScoresKey);
  auto bboxes = tracker_output.at(kBBoxesKey);

  auto ids_1d = ids.reshape(-1);
  int64_t num_tracks = ids_1d.size(0);
  TORCH_CHECK(
      num_tracks <= max_tracks_,
      "Active track count (",
      num_tracks,
      ") exceeds configured max_tracks (",
      max_tracks_,
      ")");

  auto id_options = at::TensorOptions().dtype(at::kLong).device(device());
  auto label_options = at::TensorOptions().dtype(labels.scalar_type()).device(device());
  auto score_options = at::TensorOptions().dtype(scores.scalar_type()).device(device());
  auto bbox_options = at::TensorOptions().dtype(bboxes.scalar_type()).device(device());

  auto padded_ids = at::full({max_tracks_}, -1, id_options);
  auto padded_labels = at::zeros({max_tracks_}, label_options);
  auto padded_scores = at::zeros({max_tracks_}, score_options);
  auto padded_bboxes = at::zeros({max_tracks_, 4}, bbox_options);

  copy_prefix(padded_ids, ids_1d, num_tracks);
  copy_prefix(padded_labels, labels.reshape({-1}), num_tracks);
  copy_prefix(padded_scores, scores.reshape({-1}), num_tracks);
  copy_prefix(padded_bboxes, bboxes.reshape({-1, 4}), num_tracks);

  auto long_options = at::TensorOptions().dtype(at::kLong).device(device());
  data[kIdsKey] = padded_ids;
  data[kLabelsKey] = padded_labels;
  data[kScoresKey] = padded_scores;
  data[kBBoxesKey] = padded_bboxes;
  data[kNumTracksKey] =
      at::full({1}, num_tracks, long_options);
  data[kNumDetectionsKey] =
      at::full({1}, num_detections, long_options);

  return data;
}

at::Tensor BYTETrackerCudaStatic::mask_indices(const at::Tensor& mask) const {
  if (!mask.defined() || mask.numel() == 0) {
    return at::empty({0}, mask.options().dtype(at::kLong));
  }
  if (mask.is_cuda()) {
    return build_index_tensor_cuda(mask);
  }
  return build_index_tensor_cpu(mask);
}

} // namespace tracker
} // namespace hm
