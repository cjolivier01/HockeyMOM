#pragma once

#include "hockeymom/csrc/common/JobRunner.h"
#include "hockeymom/csrc/common/MatrixRGB.h"
#include "hockeymom/csrc/mblend/mblend.h"
#include "hockeymom/csrc/pytorch/image_remap.h"
#include "hockeymom/csrc/stitcher/HmNona.h"

#include "hockeymom/csrc/mblend/threadpool.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ATen/ATen.h>

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
  struct TorchImage {
    at::Tensor tensor;
    std::vector<int> xy_pos;
  };

  static constexpr std::size_t kInvalidFrameId =
      std::numeric_limits<std::size_t>::max();
  std::size_t frame_id{kInvalidFrameId};
  // pytorch specific
  std::vector<TorchImage> torch_input_images;
  std::vector<TorchImage> torch_remapped_images;
  // Non-pytorch
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
 public:
  using FRAME_DATA_TYPE = std::shared_ptr<FrameData>;

  StitchingDataLoader(
      std::size_t start_frame_id,
      std::string project_file,
      std::string seam_file,
      std::string xor_mask_file,
      bool save_seam_and_xor_mask,
      std::size_t max_queue_size,
      std::size_t remap_thread_count,
      std::size_t blend_thread_count);
  ~StitchingDataLoader();

  void configure_remapper(std::vector<ops::RemapperConfig> remapper_configs);

  void add_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

  void add_torch_frame(
      std::size_t frame_id,
      at::Tensor images_1,
      std::vector<int> xy_pos_1,
      at::Tensor images_2,
      std::vector<int> xy_pos_2);

  void add_remapped_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

  std::shared_ptr<MatrixRGB> get_stitched_frame(std::size_t frame_id);

 private:
  void initialize();
  const std::shared_ptr<HmNona>& get_nona_worker(std::size_t worker_number);

  using JobRunnerT = JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE>;

  FRAME_DATA_TYPE remap_worker(
      std::size_t worker_index,
      FRAME_DATA_TYPE&& frame);
  FRAME_DATA_TYPE blend_worker(
      std::size_t worker_index,
      FRAME_DATA_TYPE&& frame);

  void shutdown();
  std::string project_file_;
  bool save_seam_and_xor_mask_;
  std::string seam_file_;
  std::string xor_mask_file_;
  std::size_t max_queue_size_;
  std::size_t next_frame_id_;
  std::size_t remap_thread_count_;
  std::size_t blend_thread_count_;
  std::shared_ptr<JobRunnerT::InputQueue> input_queue_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> remap_runner_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> blend_runner_;
  std::mutex nonas_create_mu_;
  std::vector<std::shared_ptr<HmNona>> nonas_;
  std::vector<std::shared_ptr<ops::ImageRemapper>> remappers_;
  std::shared_ptr<enblend::EnBlender> enblender_;
  std::vector<ops::RemapperConfig> remapper_configs_;
  std::unique_ptr<Eigen::ThreadPool> thread_pool_;
  std::unique_ptr<HmThreadPool> remap_thread_pool_;
};

} // namespace hm
