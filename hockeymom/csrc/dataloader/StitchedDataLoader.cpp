#include "hockeymom/csrc/dataloader/StitchedDataLoader.h"

namespace hm {

StitchingDataLoader::StitchingDataLoader(
    std::size_t start_frame_id,
    std::size_t thread_count)
    : next_frame_id_(start_frame_id),
      thread_count_(thread_count),
      input_queue_(
          std::make_unique<
              moodycamel::BlockingConcurrentQueue<std::unique_ptr<FrameData>>>(
              /*capacity=*/kInputQueueCapacity)) {}

StitchingDataLoader::~StitchingDataLoader() {
  shutdown();
}

void StitchingDataLoader::shutdown() {}

void StitchingDataLoader::add_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {}

} // namespace hm
