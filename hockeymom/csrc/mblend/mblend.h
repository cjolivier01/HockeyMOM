#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

#include "absl/synchronization/mutex.h"

#include <mutex>
#include <string>
#include <vector>

namespace hm {

class Blender;
class Image;

struct BlenderImageState {
  void init_from_images(
      std::vector<std::reference_wrapper<hm::MatrixRGB>> incoming_images);

  void init_from_image_state(
      const BlenderImageState& prev_image_state,
      const std::vector<std::reference_wrapper<hm::MatrixRGB>>&
          incoming_images);

  // Have to make this a shared pointer because of its forward declaration
  std::vector<std::shared_ptr<Image>> images;
};

namespace enblend {
int enblend_main(
    std::string output_image,
    std::vector<std::string> input_files);

std::unique_ptr<MatrixRGB> enblend(MatrixRGB& image1, MatrixRGB& image2);

class EnBlender {
 public:
  EnBlender(std::vector<std::string> args = {});

  std::unique_ptr<MatrixRGB> blend_images(
      const std::vector<std::shared_ptr<MatrixRGB>>& images);

 private:
  absl::Mutex mu_;
  std::shared_ptr<Blender> blender_;
  BlenderImageState image_state_;
  std::size_t pass_{0};
};

} // namespace enblend
} // namespace hm
