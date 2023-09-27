#pragma once

#include "hockeymom/csrc/common/MatrixRGB.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace hm {

/**
 *  ______                          _____        _
 * |  ____|                        |  __ \      | |
 * | |__ _ __  __ _ _ __ ___   ___ | |  | | __ _| |_  __ _
 * |  __| '__|/ _` | '_ ` _ \ / _ \| |  | |/ _` | __|/ _` |
 * | |  | |  | (_| | | | | | |  __/| |__| | (_| | |_| (_| |
 * |_|  |_|   \__,_|_| |_| |_|\___||_____/ \__,_|\__|\__,_|
 *
 */
struct FrameData {
  static constexpr std::size_t kInvalidFrameId =
      std::numeric_limits<std::size_t>::max();
  std::size_t frame_id{kInvalidFrameId};
  std::vector<std::shared_ptr<MatrixRGB>> input_images;
  std::vector<std::shared_ptr<MatrixRGBA>> remapped_images;
  std::vector<std::shared_ptr<MatrixRGBA>> blended_images;
};

/**
 *   _____            _             _  ____
 *  / ____|          | |           | |/ __ \
 * | (___   ___  _ __| |_  ___   __| | |  | |_   _  ___  _   _  ___
 *  \___ \ / _ \| '__| __|/ _ \ / _` | |  | | | | |/ _ \| | | |/ _ \
 *  ____) | (_) | |  | |_|  __/| (_| | |__| | |_| |  __/| |_| |  __/
 * |_____/ \___/|_|   \__|\___| \__,_|\___\_\\__,_|\___| \__,_|\___|
 *
 */
template <typename KEY_TYPE, typename ITEM_TYPE>
class SortedQueue {
 public:
  SortedQueue() = default;

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

  ITEM_TYPE dequeue_key(const KEY_TYPE& key) {
    std::unique_lock lk(items_mu_);
    cu_.wait(lk, [this, &key] { return items_.find(key) != items_.end(); });
    auto iter = items_.find(key);
    assert(iter != items_.end());
    auto value = std::move(iter->second);
    items_.erase(iter);
    return value;
  }

 private:
  std::mutex items_mu_;
  std::condition_variable cu_;
  std::map<KEY_TYPE, ITEM_TYPE> items_;
};

/**
 *       _       _     _____
 *      | |     | |   |  __ \
 *      | | ___ | |__ | |__) |_   _ _ __  _ __   ___  _ __
 *  _   | |/ _ \| '_ \|  _  /| | | | '_ \| '_ \ / _ \| '__|
 * | |__| | (_) | |_) | | \ \| |_| | | | | | | |  __/| |
 *  \____/ \___/|_.__/|_|  \_\\__,_|_| |_|_| |_|\___||_|
 *
 */
template <typename INPUT_TYPE, typename OUTPUT_TYPE>
class JobRunner {
 public:
  using KeyType = std::size_t;
  using InputQueue = SortedQueue<KeyType, INPUT_TYPE>;
  using OutputQueue = SortedQueue<KeyType, INPUT_TYPE>;

  JobRunner(
      std::shared_ptr<InputQueue> input_queue,
      std::function<OUTPUT_TYPE(INPUT_TYPE&&)> worker_fn)
      : input_queue_(std::move(input_queue)),
        worker_fn_(std::move(worker_fn_)) {}
  ~JobRunner() {
    stop();
  }

  void start(std::size_t thread_count) {
    stop();
    threads_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back(
          std::make_unique<std::thread>([this, i] { this->run(i); }));
    }
  }

  void stop() {
    for (std::size_t i = 0, n = threads_.size(); i < n; ++i) {
      INPUT_TYPE null_input;
      assert(!null_input);
      input_queue_->enqueue(
          std::numeric_limits<KeyType>::max() - i - 1, std::move(null_input));
    }
    for (auto& t : threads_) {
      t->join();
    }
    threads_.clear();
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
      KeyType key{std::numeric_limits<KeyType>::max()};
      INPUT_TYPE input = input_queue_->dequeue(&key);
      if (!input) {
        break;
      }
      output_queue_->enqueue(key, worker_fn_(std::move(input)));
    } while (true);
  }
  std::shared_ptr<InputQueue> input_queue_;
  std::shared_ptr<OutputQueue> output_queue_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::function<OUTPUT_TYPE(INPUT_TYPE&&)> worker_fn_;
};

/* clang-format off */
/**
 *   _____ _   _  _        _     _             _____        _         _                      _
 *  / ____| | (_)| |      | |   (_)           |  __ \      | |       | |                    | |
 * | (___ | |_ _ | |_  ___| |__  _ _ __   __ _| |  | | __ _| |_  __ _| |      ___   __ _  __| | ___  _ __
 *  \___ \| __| || __|/ __| '_ \| | '_ \ / _` | |  | |/ _` | __|/ _` | |     / _ \ / _` |/ _` |/ _ \| '__|
 *  ____) | |_| || |_| (__| | | | | | | | (_| | |__| | (_| | |_| (_| | |____| (_) | (_| | (_| |  __/| |
 * |_____/ \__|_| \__|\___|_| |_|_|_| |_|\__, |_____/ \__,_|\__|\__,_|______|\___/ \__,_|\__,_|\___||_|
 *                                        __/ |
 *                                       |___/
 */
/* clang-format on */
class StitchingDataLoader {
  static constexpr std::size_t kInputQueueCapacity = 32;

 public:
  using FRAME_DATA_TYPE = std::shared_ptr<FrameData>;

  StitchingDataLoader(
      std::size_t start_frame_id,
      std::size_t remap_thread_count,
      std::size_t blend_thread_count);
  ~StitchingDataLoader();

  void add_frame(
      std::size_t frame_id,
      std::vector<std::shared_ptr<MatrixRGB>>&& images);

 private:
  using JobRunnerT = JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE>;

  FRAME_DATA_TYPE remap_worker(FRAME_DATA_TYPE&& frame);
  FRAME_DATA_TYPE blend_worker(FRAME_DATA_TYPE&& frame);

  void shutdown();

  std::size_t next_frame_id_;
  std::size_t remap_thread_count_;
  std::size_t blend_thread_count_;
  std::shared_ptr<JobRunnerT::InputQueue> input_queue_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> remap_runner_;
  JobRunner<FRAME_DATA_TYPE, FRAME_DATA_TYPE> blend_runner_;
};

} // namespace hm
