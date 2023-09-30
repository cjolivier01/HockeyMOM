#pragma once

#include "hockeymom/csrc/common/JobRunner.h"
#include "hockeymom/csrc/common/MatrixRGB.h"
#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/stitcher/HmNona.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hm {

/**
 *  ______                          _____        _
 * |  ____|                        |  __ \      | |
 * | |__ _ __  __ _ _ __ ___   ___ | |  | | __ _| |_  __ _
 * |  __| '__|/ _` | '_ ` _ \ / _ \| |  | |/ _` | __|/ _` |
 * | |  | |  | (_| | | | | | |  __/| |__| | (_| | |_| (_| |
 * |_|  |_|   \__,_|_| |_| |_|\___||_____/ \__,_|\__|\__,_|
 *
 */
struct FrameData {
  static constexpr std::size_t kInvalidFrameId =
      std::numeric_limits<std::size_t>::max();
  std::size_t frame_id{kInvalidFrameId};
  std::vector<std::shared_ptr<MatrixRGB>> input_images;
  std::vector<std::shared_ptr<MatrixRGB>> remapped_images;
  std::shared_ptr<MatrixRGB> blended_image;
};

/* clang-format off */
/**
 *   _____ _   _  _        _     _             _____        _         _                      _
 *  / ____| | (_)| |      | |   (_)           |  __ \      | |       | |                    | |
 * | (___ | |_ _ | |_  ___| |__  _ _ __   __ _| |  | | __ _| |_  __ _| |      ___   __ _  __| | ___  _ __
 *  \___ \| __| || __|/ __| '_ \| | '_ \ / _` | |  | |/ _` | __|/ _` | |     / _ \ / _` |/ _` |/ _ \| '__|
 *  ____) | |_| || |_| (__| | | | | | | | (_| | |__| | (_| | |_| (_| | |____| (_) | (_| | (_| |  __/| |
 * |_____/ \__|_| \__|\___|_| |_|_|_| |_|\__, |_____/ \__,_|\__|\__,_|______|\___/ \__,_|\__,_|\___||_|
 *                                        __/ |
 *                                       |___/
 */
/* clang-format on */
class StitchingDataLoader {
  static constexpr std::size_t kInputQueueCapacity = 32;

 public:
  using FRAME_DATA_TYPE = std::shared_ptr<FrameData>;

  StitchingDataLoader(
      std::size_t start_frame_id,
      std::string project_file,
      std::size_t max_queue_size,
      std::size_t remap_thread_count,
      std::size_t blend_thread_count);
  ~StitchingDataLoader();

  void add_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

  std::shared_ptr<MatrixRGB> get_stitched_frame(std::size_t frame_id);

 private:
  void initialize();

  using JobRunnerT = JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE>;

  FRAME_DATA_TYPE remap_worker(
      std::size_t worker_index,
      FRAME_DATA_TYPE&& frame);
  FRAME_DATA_TYPE blend_worker(
      std::size_t worker_index,
      FRAME_DATA_TYPE&& frame);

  void shutdown();
  std::string project_file_;
  std::size_t max_queue_size_;
  std::size_t next_frame_id_;
  std::size_t remap_thread_count_;
  std::size_t blend_thread_count_;
  std::shared_ptr<JobRunnerT::InputQueue> input_queue_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> remap_runner_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> blend_runner_;
  std::vector<std::shared_ptr<HmNona>> nonas_;
  //std::vector<std::shared_ptr<enblend::EnBlender>> enblenders_;
  std::shared_ptr<enblend::EnBlender> enblender_;
};

} // namespace hm
