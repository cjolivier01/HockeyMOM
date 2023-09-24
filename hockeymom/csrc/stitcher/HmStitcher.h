#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"
#include "hockeymom/csrc/stitcher/HmRemappedPanoImage.h"

#include "nona/StitcherOptions.h"
#include "panodata/Panorama.h"

#include <cstdint>
#include <string>

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
 public:
  HmNona(std::string project_file);
  ~HmNona();
  std::size_t count() const {
    return 0;
  }
  bool load_project(const std::string& project_file);
  std::pair<std::unique_ptr<hm::MatrixRGB>, std::unique_ptr<hm::MatrixRGB>>
  process_images(
      std::shared_ptr<hm::MatrixRGB> image1,
      std::shared_ptr<hm::MatrixRGB> image2);

 private:
  std::string project_file_;
  HuginBase::PanoramaOptions opts_;
  HuginBase::Nona::AdvancedOptions adv_options_;
  HuginBase::Panorama pano_;
};

} // namespace hm
