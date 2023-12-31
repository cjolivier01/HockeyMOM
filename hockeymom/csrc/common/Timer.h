#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>

namespace hm {

class Timer {
 public:
  Timer();
  void tic() {
    assert(!in_tic_);
    start_time_ = std::chrono::high_resolution_clock::now();
    in_tic_ = true;
  }
  void toc() {
    auto end_time = std::chrono::high_resolution_clock::now();
    total_microseconds_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_)
            .count();
    ++count_;
    in_tic_ = false;
  }
  double average_time() const {
    if (!count_) {
      return 0.0;
    }
    return double(total_microseconds_) / double(count_) / 1000000.0;
  }
  double fps() const {
    if (!count_) {
      return 0.0;
    }
    auto average_t = average_time();
    return 1.0 / average_t;
  }
  void reset() {
    count_ = 0;
    total_microseconds_ = 0;
    in_tic_ = false;
  }

 private:
  std::size_t count_{0};
  std::size_t total_microseconds_{0};
  std::chrono::high_resolution_clock::time_point start_time_;
  bool in_tic_{false};
};

} // namespace hm