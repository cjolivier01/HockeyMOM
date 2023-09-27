#pragma once

#include "hugin_base/hugin_utils/utils.h"

#include <string>

namespace hm {

bool check_cuda_opengl();

std::size_t get_tick_count_ms();

void set_thread_name(const std::string& thread_name, int index = -1);

}  // namespace hm
