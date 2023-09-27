#include "hockeymom/csrc/dataloader/StitchedDataLoader.h"
#include "hockeymom/csrc/stitcher/HmNona.h"
#include "hockeymom/csrc/mblend/mblend.h"

namespace hm {

StitchingDataLoader::StitchingDataLoader(
    std::size_t start_frame_id,
    std::size_t remap_thread_count,
    std::size_t blend_thread_count)
    : next_frame_id_(start_frame_id),
      remap_thread_count_(remap_thread_count),
      blend_thread_count_(blend_thread_count),
      input_queue_(std::make_shared<JobRunnerT::InputQueue>()),
      remap_runner_(
          input_queue_,
          [this](StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->remap_worker(std::move(frame));
          }),
      blend_runner_(
          remap_runner_.outputs(),
          [this](StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->blend_worker(std::move(frame));
          }) {
  remap_runner_.start(remap_thread_count_);
  blend_runner_.start(blend_thread_count_);
}

StitchingDataLoader::~StitchingDataLoader() {
  shutdown();
}

void StitchingDataLoader::shutdown() {
  remap_runner_.stop();
  blend_runner_.stop();
}

void StitchingDataLoader::add_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {
  auto frame_info = std::make_shared<FrameData>();
  frame_info->frame_id = frame_id;
  frame_info->input_images = std::move(images);
  remap_runner_.inputs()->enqueue(frame_id, std::move(frame_info));
}

StitchingDataLoader::FRAME_DATA_TYPE remap_worker(
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  return frame;
}

StitchingDataLoader::FRAME_DATA_TYPE blend_worker(
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  return frame;
}

} // namespace hm
