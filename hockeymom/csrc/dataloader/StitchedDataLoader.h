#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

//#include "concurrentqueue/blockingconcurrentqueue.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hm {

struct FrameData {
  static constexpr std::size_t kInvalidFrameId =
      std::numeric_limits<std::size_t>::max();
  std::size_t frame_id{kInvalidFrameId};
  std::vector<std::shared_ptr<MatrixRGB>> input_images;
  std::vector<std::shared_ptr<MatrixRGBA>> remapped_images;
  std::vector<std::shared_ptr<MatrixRGBA>> blended_images;
};

template <typename KEY_TYPE, typename ITEM_TYPE>
class SortedQueue {
 public:
  SortedQueue();
  void enqueue(const KEY_TYPE& key_type, ITEM_TYPE&& item) {
    std::unique_lock lk(items_mu_);
    if (!items_.emplace(key_type, std::move(item)).second) {
      throw std::runtime_error("Duplciate key in sorted queue");
    }
    cu_.notify_one();
  }
  ITEM_TYPE dequeue(KEY_TYPE* key_type_ptr = nullptr) {
    std::unique_lock lk(items_mu_);
    cu_.wait(lk, [this] { return !items_.empty(); });
    auto iter = items_.begin();
    auto value = std::move(iter->second);
    if (key_type_ptr) {
      *key_type_ptr = iter->first;
    }
    items_.erase(iter);
    return value;
  }

 private:
  std::mutex items_mu_;
  std::condition_variable cu_;
  std::map<KEY_TYPE, ITEM_TYPE> items_;
};

template <typename INPUT_TYPE, typename OUTPUT_TYPE>
class JobRunner {
 public:
  using KeyType = std::size_t;
  using InputQueue = SortedQueue<KeyType, INPUT_TYPE>;
  using OutputQueue = SortedQueue<KeyType, INPUT_TYPE>;

  JobRunner(
      std::size_t thread_count,
      std::shared_ptr<InputQueue> input_queue,
      std::function<OUTPUT_TYPE(INPUT_TYPE&&)> worker_fn)
      : input_queue_(std::move(input_queue)),
        worker_fn_(std::move(worker_fn_)) {
    start();
  }
  ~JobRunner() {
    stop();
  }

  const std::shared_ptr<InputQueue>& inputs() {
    return input_queue_;
  }
  const std::shared_ptr<OutputQueue>& outputs() {
    return output_queue_;
  }

 private:
  void run(std::size_t thread_id) {
    do {
      INPUT_TYPE input;
      input_queue_->wait_dequeue(input);
      if (!input) {
        break;
      }
      output_queue_->enqueue(worker_fn_(std::move(input)));
    } while (true);
  }
  void start(std::size_t thread_count) {
    stop();
    threads_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back(std::make_unique<std::thread>(run, this, i));
    }
  }
  void stop() {
    for (std::size_t i = 0, n = threads_.size(); i < n; ++i) {
      INPUT_TYPE null_input;
      assert(!null_input);
      input_queue_->enqueue(null_input);
    }
    for (auto& t : threads_) {
      t->join();
    }
    threads_.clear();
  }
  std::shared_ptr<InputQueue> input_queue_;
  std::shared_ptr<OutputQueue> output_queue_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::function<OUTPUT_TYPE(INPUT_TYPE&&)> worker_fn_;
};

class StitchingDataLoader {
  static constexpr std::size_t kInputQueueCapacity = 32;
  using FRAME_DATA_TYPE = std::shared_ptr<FrameData>;
  using JobRunnerT = JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE>;
 public:
  StitchingDataLoader(std::size_t start_frame_id, std::size_t remap_thread_count, std::size_t blend_thread_count);
  ~StitchingDataLoader();

  void add_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

 private:
  void shutdown();

  std::size_t next_frame_id_;
  std::size_t remap_thread_count_;
  std::size_t blend_thread_count_;
  std::shared_ptr<JobRunnerT::InputQueue> input_queue_;
};

} // namespace hm
