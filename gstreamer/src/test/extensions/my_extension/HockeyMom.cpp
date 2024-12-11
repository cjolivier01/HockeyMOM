
#include "HockeyMom.hpp"  // NOLINT

namespace sample {
namespace test {

gxf_result_t HockeyMom::start() {

    GXF_LOG_INFO("HockeyMom::start");
    return GXF_SUCCESS;
}

gxf_result_t HockeyMom::tick() {
  GXF_LOG_INFO("HockeyMom::tick");
  return GXF_SUCCESS;
}

gxf_result_t HockeyMom::stop() { 
  GXF_LOG_INFO("HockeyMom::stop");
  return GXF_SUCCESS;
}

}  // namespace test
}  // namespace sample

  