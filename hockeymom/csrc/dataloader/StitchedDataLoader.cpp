#include "hockeymom/csrc/dataloader/StitchedDataLoader.h"

namespace hm {

StitchingDataLoader::StitchingDataLoader(
    std::size_t start_frame_id, std::size_t remap_thread_count, std::size_t blend_thread_count)
    : next_frame_id_(start_frame_id),
      remap_thread_count_(remap_thread_count),
      blend_thread_count_(blend_thread_count),
      input_queue_(std::make_shared<JobRunnerT::InputQueue>()) {}

StitchingDataLoader::~StitchingDataLoader() {
  shutdown();
}

void StitchingDataLoader::shutdown() {}

void StitchingDataLoader::add_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {}

} // namespace hm
