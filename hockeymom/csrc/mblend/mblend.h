#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

#include <string>
#include <vector>

namespace hm {

class Blender;

namespace enblend {
int enblend_main(
    std::string output_image,
    std::vector<std::string> input_files);

std::unique_ptr<MatrixRGB> enblend(MatrixRGB& image1, MatrixRGB& image2);

class EnBlender {
  public:
  EnBlender(std::vector<std::string> args = {});

  std::unique_ptr<MatrixRGB> blend_images(const std::vector<std::shared_ptr<MatrixRGB>>& images);

  private:
    std::shared_ptr<Blender> blender_;
    std::size_t pass_{0};
};

} // namespace enblend
} // namespace hm
