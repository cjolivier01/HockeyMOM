#pragma once

#include "hockeymom/csrc/common/Gpu.h"

#include <cassert>
#include <iostream>
#include <thread>
#include <limits>
#include "threadpool.h"

namespace hm {

/*static*/ std::atomic<std::size_t> HmThreadPool::next_job_id_{1};
/* clang-format on */

HmThreadPool::HmThreadPool(Eigen::ThreadPool* thread_pool)
    : thread_pool_(thread_pool) {}

HmThreadPool::HmThreadPool(const HmThreadPool& thread_pool_group)
    : thread_pool_(thread_pool_group.get_internal_thread_pool()) {}

HmThreadPool& HmThreadPool::operator=(Eigen::ThreadPool* thread_pool) {
  thread_pool_ = thread_pool;
  return *this;
}

HmThreadPool::~HmThreadPool() {
  join_all(/*allow_reuse=*/false);
}

// std::size_t thread_local HmThreadPool::current_thread_pool_thread_id_{
//     std::numeric_limits<std::size_t>::max()} /*NOLINT*/;

// std::size_t HmThreadPool::GetCurrentThreadLocalId() {
//   if (current_thread_pool_thread_id_ ==
//       std::numeric_limits<std::size_t>::max()) {
//     return get_current_thread_id();
//   }
//   return current_thread_pool_thread_id_;
// }

// std::size_t HmThreadPool::CurrentThreadId() const {
//   return thread_pool_->CurrentThreadId();
// }

HmThreadPool::operator bool() const {
  return thread_pool_ != nullptr;
}

void HmThreadPool::join(std::size_t job_id) {
  // We don't need the schedule lock here
  using Args = std::tuple<HmThreadPool*, std::size_t>;
  Args args{this, job_id};
  absl::MutexLock lk_inner(&mu_);
  mu_.Await(absl::Condition(
      +[](Args* args_ptr)
           ABSL_EXCLUSIVE_LOCKS_REQUIRED(std::get<0>(*args_ptr)->mu_) {
             return !std::get<0>(*args_ptr)->active_job_ids_.count(
                 std::get<1>(*args_ptr));
           },
      &args));
}

bool HmThreadPool::try_join(std::size_t job_id, absl::Duration timeout) {
  // We don't need the schedule lock here
  using Args = std::tuple<HmThreadPool*, std::size_t>;
  Args args{this, job_id};
  absl::MutexLock lk_inner(&mu_);
  if (!mu_.AwaitWithTimeout(
          absl::Condition(
              +[](Args* args_ptr)
                   ABSL_EXCLUSIVE_LOCKS_REQUIRED(std::get<0>(*args_ptr)->mu_) {
                     return !std::get<0>(*args_ptr)->active_job_ids_.count(
                         std::get<1>(*args_ptr));
                   },
              &args),
          timeout)) {
    return false;
  }
  return true;
}

void HmThreadPool::join_all(bool allow_reuse) {
  // Block anything getting scheduled while we perform a join-all
  absl::MutexLock lk(&schedule_mutex_);
  absl::MutexLock lk_inner(&mu_);
  mu_.Await(absl::Condition(
      +[](HmThreadPool* this_ptr) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
           this_ptr->mu_) { return this_ptr->active_job_ids_.empty(); },
      this));
  if (!allow_reuse) {
    final_join_called_ = true;
  }
}

bool HmThreadPool::try_join_all(absl::Duration timeout, bool allow_reuse) {
  // Block anything getting scheduled while we perform a try-join-all
  absl::MutexLock lk(&schedule_mutex_);
  absl::MutexLock lk_inner(&mu_);
  if (!mu_.AwaitWithTimeout(
          absl::Condition(
              +[](HmThreadPool* this_ptr) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
                   this_ptr->mu_) { return this_ptr->active_job_ids_.empty(); },
              this),
          timeout)) {
    return false;
  }
  if (!allow_reuse) {
    final_join_called_ = true;
  }
  return true;
}

void HmThreadPool::final_join_all() {
  join_all(/*allow_reuse=*/false);
}

std::size_t HmThreadPool::Schedule(std::function<void()>&& thread_fn) {
  std::size_t new_id = next_job_id_++;

  absl::MutexLock lk_sched(&schedule_mutex_);
  {
    absl::MutexLock lk(&mu_);
    active_job_ids_.emplace(new_id);
  }
  thread_pool_->Schedule([this, new_id, fn = std::move(thread_fn)]() {
    // Save the thread-local ID and set the new one from the pool's thread id
    // function
    //auto save_thread_id = current_thread_pool_thread_id_;
    //current_thread_pool_thread_id_ = CurrentThreadId();
    //set_thread_name("tg_pool_job", new_id);
    try {
      fn();
    } catch (...) {
      //  The idea here isn't to try to "save" whatever went wrong (or whatever
      //  threw an exception) within the job function, but simply to guarantee
      //  even if something goes badly, it won't cause the join_all() to block
      //  forever because the job id was never removed from the active_job_ids_
      //  structure. After we remove it the job id, we just rethrow it and let
      //  the thread pool sort it out (maybe it assers, but if we're trying to
      //  make a clean shutdown, from the ensuing SIGABRT, for instance, then we
      //  don't want to get hung up in the HmThreadPool's join_all())
      absl::MutexLock lk(&mu_);
      active_job_ids_.erase(new_id);
      // restore the thread_local current thread id var
      //current_thread_pool_thread_id_ = save_thread_id;
      throw;
    }
    absl::MutexLock lk(&mu_);
    active_job_ids_.erase(new_id);
    // restore the thread_local current thread id var
    //current_thread_pool_thread_id_ = save_thread_id;
  });
  return new_id;
}


// namespace {
// std::mutex tp_mu;
// }

// //std::unique_ptr<ThreadPool> ThreadPool::instance{nullptr};
// ThreadPool* ThreadPool::instance{nullptr};

// /*static*/
// ThreadPool* ThreadPool::GetInstance(int threads) {
//   std::unique_lock<std::mutex> lk(tp_mu);
//   if (!instance) {
//     //    instance = std::make_unique<ThreadPool>(threads);
//     instance = new ThreadPool(threads);
//   }
//   return instance;
// }

// /**********************************************************************
//  * Constructor (private)
//  **********************************************************************/
// ThreadPool::ThreadPool(int _threads) {
//   n_threads = _threads > 0
//       ? std::min((unsigned int)_threads, std::thread::hardware_concurrency())
//       : std::thread::hardware_concurrency();
//   threads.resize(n_threads);
//   // std::cout << "Creating thread pool..." << std::endl;
//   for (int i = 0; i < n_threads; ++i) {
//     threads[i].main_mutex = &main_mutex;
//     threads[i].return_mutex = &return_mutex;
//     threads[i].main_cond = &main_cond;
//     threads[i].return_cond = &return_cond;
//     threads[i].free = true;
//     threads[i].stop = false;
//     threads[i].queue = &queue;
//     threads[i].i = i;
// #ifdef _WIN32
//     threads[i].handle = CreateThread(
//         NULL, 1, (LPTHREAD_START_ROUTINE)Thread, &threads[i], 0, NULL);
// #else
//     pthread_create(&threads[i].handle, NULL, TP_Thread, &threads[i]);
// #endif
//   }
// }

// /**********************************************************************
//  * Destructor
//  **********************************************************************/
// ThreadPool::~ThreadPool() {
//   int i;
//   // std::cout << "Destroying thread pool..." << std::endl;
//   {
//     std::lock_guard<std::mutex> mlock(main_mutex);
//     for (i = 0; i < n_threads; ++i) {
//       threads[i].stop = true;
//     }
//   }

//   main_cond.notify_all();
//   for (i = 0; i < n_threads; ++i) {
// #ifdef _WIN32
//     WaitForSingleObject(threads[i].handle, INFINITE);
// #else
//     pthread_join(threads[i].handle, NULL);
// #endif
//   }
// }

// /**********************************************************************
//  * Threads
//  **********************************************************************/
// #define P ((ThreadPool::tp_struct*)param)
// static std::atomic<std::size_t> tpool_thread_nr{0};
// #ifdef _WIN32
// DWORD WINAPI ThreadPool::Thread(void* param) {
// #else
// void* TP_Thread(void* param) {
// #endif
//   set_thread_name("t-pool", tpool_thread_nr++);
//   while (true) {
//     {
//       std::unique_lock<std::mutex> mlock(*P->main_mutex);
//       P->main_cond->wait(mlock, [=] { return P->queue->size() || P->stop; });
//       if (P->queue->size()) {
//         P->function = P->queue->front();
//         P->queue->pop_front();
//         P->free = false;
//       }
//     }
//     if (P->stop)
//       break;

//     P->function();

//     {
//       std::lock_guard<std::mutex> mlock(*P->return_mutex); // necessary
//       P->free = true;
//     }
//     P->return_cond->notify_all();
//   }

//   return 0;
// }

// /**********************************************************************
//  * Wait
//  **********************************************************************/
// void ThreadPool::Wait() {
//   if (!queue.size()) {
//     int i;
//     for (i = 0; i < n_threads; ++i) {
//       if (!threads[i].free)
//         break;
//     }
//     if (i == n_threads)
//       return;
//   }

//   {
//     std::unique_lock<std::mutex> rlock(return_mutex);
//     return_cond.wait(rlock, [=] {
//       if (queue.size())
//         return false;
//       for (int i = 0; i < n_threads; ++i) {
//         if (!threads[i].free) {
//           return false;
//         }
//       }
//       return true;
//     });
//   }
// }

// /**********************************************************************
//  * Queue
//  **********************************************************************/
// void ThreadPool::Queue(std::function<void()> function) {
//   std::lock_guard<std::mutex> mlock(main_mutex); // not sure what this is for
//   queue.push_back(std::move(function));
//   main_cond.notify_one(); // changed from notify_all()
// }
} // namespace hm
