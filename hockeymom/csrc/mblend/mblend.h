#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

#include <string>
#include <vector>

namespace hm {
namespace enblend {
int enblend_main(
    std::string output_image,
    std::vector<std::string> input_files);

std::unique_ptr<MatrixRGB> enblend(MatrixRGB& image1, MatrixRGB& image2);

} // namespace enblend
} // namespace hm
