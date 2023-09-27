#include "hockeymom/csrc/dataloader/StitchingDataLoader.h"
#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/stitcher/HmNona.h"

namespace hm {

StitchingDataLoader::StitchingDataLoader(
    std::size_t start_frame_id,
    std::string project_file,
    std::size_t remap_thread_count,
    std::size_t blend_thread_count)
    : project_file_(std::move(project_file)),
      next_frame_id_(start_frame_id),
      remap_thread_count_(remap_thread_count),
      blend_thread_count_(blend_thread_count),
      input_queue_(std::make_shared<JobRunnerT::InputQueue>()),
      remap_runner_(
          input_queue_,
          [this](
              std::size_t worker_index,
              StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->remap_worker(worker_index, std::move(frame));
          }),
      blend_runner_(
          remap_runner_.outputs(),
          [this](
              std::size_t worker_index,
              StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
            return this->blend_worker(worker_index, std::move(frame));
          }) {
  initialize();
}

StitchingDataLoader::~StitchingDataLoader() {
  shutdown();
}

void StitchingDataLoader::shutdown() {
  remap_runner_.stop();
  blend_runner_.stop();
  nonas_.clear();
}

void StitchingDataLoader::initialize() {
  assert(nonas_.empty());
  nonas_.resize(remap_thread_count_);
  remap_runner_.start(remap_thread_count_);
  blend_runner_.start(blend_thread_count_);
}

void StitchingDataLoader::add_frame(
    std::size_t frame_id,
    std::vector<std::shared_ptr<MatrixRGB>>&& images) {
  auto frame_info = std::make_shared<FrameData>();
  frame_info->frame_id = frame_id;
  frame_info->input_images = std::move(images);
  remap_runner_.inputs()->enqueue(frame_id, std::move(frame_info));
}

std::shared_ptr<MatrixRGB> StitchingDataLoader::get_stitched_frame(
    std::size_t frame_id) {
  auto final_frame = blend_runner_.outputs()->dequeue_key(frame_id);
  return final_frame->blended_image;
}

StitchingDataLoader::FRAME_DATA_TYPE StitchingDataLoader::remap_worker(
    std::size_t worker_index,
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  if (!nonas_.at(worker_index)) {
    nonas_[worker_index] = std::make_unique<HmNona>(project_file_);
  }
  auto remapped = nonas_[worker_index]->remap_images(
      std::move(frame->input_images.at(0)),
      std::move(frame->input_images.at(1)));
  frame->remapped_images.clear();
  for (auto& r : remapped) {
    frame->remapped_images.emplace_back(std::move(r));
  }
  return frame;
}

StitchingDataLoader::FRAME_DATA_TYPE StitchingDataLoader::blend_worker(
    std::size_t worker_index,
    StitchingDataLoader::FRAME_DATA_TYPE&& frame) {
  // TEMP HACK, SEND BACK FIRST IMAGE
  frame->blended_image = frame->remapped_images.at(0);
  return frame;
}

} // namespace hm
