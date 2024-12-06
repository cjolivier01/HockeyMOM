#include "hockeymom/csrc/play_tracker/LivingBox.h"

#include <cassert>
#include <memory>

namespace hm {
namespace play_tracker {

std::unique_ptr<ILivingBox> create_live_box(
    std::string label,
    const BBox& bbox,
    const AllLivingBoxConfig* config) {
  const static AllLivingBoxConfig default_config;
  const AllLivingBoxConfig* cfg = config ? config : &default_config;
  return std::make_unique<LivingBox>(std::move(label), bbox, *cfg);
}

} // namespace play_tracker
} // namespace hm
