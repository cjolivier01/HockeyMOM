#include "hockeymom/csrc/postprocess/ImagePostProcess.h"

#include <iostream>
#include <sstream>

namespace hm {

std::string HMPostprocessConfig::to_string() const {
  std::stringstream ss;
  ss << "use_watermark = " << (use_watermark ? "true" : "false") << "\n";
  return ss.str();
}

} // namespace hm
