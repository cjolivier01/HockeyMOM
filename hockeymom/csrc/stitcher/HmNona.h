#pragma once

#include "hockeymom/csrc/common/Gpu.h"
#include "hockeymom/csrc/common/MatrixRGB.h"
#include "hockeymom/csrc/stitcher/FileRemapper.h"
#include "hockeymom/csrc/stitcher/HmStitcher.h"

#include "nona/StitcherOptions.h"
#include "panodata/Panorama.h"

#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace hm {

/**
 *  _    _           _   _
 * | |  | |         | \ | |
 * | |__| |_ __ ___ |  \| | ___  _ __   __ _
 * |  __  | '_ ` _ \| . ` |/ _ \| '_ \ / _` |
 * | |  | | | | | | | |\  | (_) | | | | (_| |
 * |_|  |_|_| |_| |_|_| \_|\___/|_| |_|\__,_|
 *
 *
 */
class HmNona {
  using ImageType = vigra::BRGBImage;

 public:
  HmNona(std::string project_file);
  ~HmNona();
  bool load_project(const std::string& project_file);

  std::vector<std::unique_ptr<hm::MatrixRGB>> remap_images(
      std::shared_ptr<hm::MatrixRGB> image1,
      std::shared_ptr<hm::MatrixRGB> image2);

 private:
  std::string project_file_;
  HuginBase::PanoramaOptions opts_;
  HuginBase::Nona::AdvancedOptions adv_options_;
  HuginBase::Panorama pano_;
  HmFileRemapper<ImageType, vigra::BImage> file_remapper_;
  std::size_t image_pair_pass_count_{0};
  std::unique_ptr<AppBase::DummyProgressDisplay> pdisp_;
  std::unique_ptr<HmMultiImageRemapper<ImageType, vigra::BImage>> stitcher_;
  
  static inline std::mutex gpu_thread_pool_mu_;
  static inline std::unique_ptr<Eigen::ThreadPool> gpu_thread_pool_;
  static std::size_t nona_count_;
};

} // namespace hm
