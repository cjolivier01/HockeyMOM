
#pragma once

#include "gxf/std/codelet.hpp"

namespace sample {
namespace test {

// Logs a message in start() and tick()
class HelloWorld : public nvidia::gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
};

}  // namespace test
}  // namespace sample
  