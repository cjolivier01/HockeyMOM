#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

#include "concurrentqueue/blockingconcurrentqueue.h"

#include <string>
#include <vector>

namespace hm {

struct FrameData {
  static constexpr std::size_t kInvalidFrameId =
      std::numeric_limits<std::size_t>::max();
  std::size_t frame_id{kInvalidFrameId};
  std::vector<std::shared_ptr<MatrixRGB>> images;
};

class StitchingDataLoader {
  static constexpr std::size_t kInputQueueCapacity = 32;
 public:
  StitchingDataLoader(std::size_t start_frame_id, std::size_t thread_count);
  ~StitchingDataLoader();

  void add_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

 private:
  void shutdown();

  std::size_t next_frame_id_{0};
  std::size_t thread_count_;
  std::unique_ptr<
      moodycamel::BlockingConcurrentQueue<std::unique_ptr<FrameData>>>
      input_queue_;
};

} // namespace hm
