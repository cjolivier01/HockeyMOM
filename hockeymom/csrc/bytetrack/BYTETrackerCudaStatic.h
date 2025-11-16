#pragma once

#include "BYTETrackerCuda.h"

#include <torch/torch.h>

namespace hm {
namespace tracker {

class BYTETrackerCudaStatic {
 public:
  explicit BYTETrackerCudaStatic(
      ByteTrackConfig config,
      int64_t max_detections,
      int64_t max_tracks,
      c10::Device device = c10::Device(c10::kCUDA, 0));

  ~BYTETrackerCudaStatic() = default;

  void reset();

  std::size_t num_tracks() const {
    return tracker_.num_tracks();
  }

  std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data);

  int64_t max_detections() const {
    return max_detections_;
  }

  int64_t max_tracks() const {
    return max_tracks_;
  }

 private:
  ByteTrackConfig config_;
  BYTETrackerCuda tracker_;
  int64_t max_detections_;
  int64_t max_tracks_;
  c10::Device device_;
};

} // namespace tracker
} // namespace hm

