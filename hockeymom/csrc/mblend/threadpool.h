#pragma once

#include "unsupported/Eigen/CXX11/ThreadPool"

#include <atomic>
//#include <condition_variable>
//#include <deque>
#include <functional>
//#include <mutex>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace hm {

// using ThreadPool = Eigen::ThreadPool;

// #ifndef _WIN32
// void* TP_Thread(void* param);
// #endif

// class ThreadPool {
// public:
// 	static ThreadPool* GetInstance(int threads = 0);
// 	void Queue(std::function<void()> function);
// 	int GetNThreads() { return n_threads; };
// 	void Wait();
// 	struct tp_struct {
// #ifdef _WIN32
// 		HANDLE handle;
// #else
// 		pthread_t handle;
// #endif
// 		std::function<void()> function;
// 		bool free;
// 		bool stop;
// 		std::mutex* main_mutex;
// 		std::mutex* return_mutex;
// 		std::condition_variable* main_cond;
// 		std::condition_variable* return_cond;
// 		std::deque<std::function<void()>>* queue;
// 		int i;
// 	};

// private:
// 	//static std::unique_ptr<ThreadPool> instance;
//   static ThreadPool *instance;
// 	ThreadPool(int _threads = 0); // constructor is private
// 	~ThreadPool();
// #ifdef _WIN32
// 	static DWORD WINAPI Thread(void* param);
// #endif
// 	std::vector<tp_struct> threads;
// 	std::deque<std::function<void()>> queue;
// 	int n_threads;
// 	std::mutex main_mutex;
// 	std::mutex return_mutex;
// 	std::condition_variable main_cond;
// 	std::condition_variable return_cond;
// };

class HmThreadPool {
 public:
  HmThreadPool(Eigen::ThreadPool* thread_pool);
  HmThreadPool(const HmThreadPool& thread_pool_group);

  ~HmThreadPool();

  HmThreadPool& operator=(Eigen::ThreadPool* thread_pool);

  /**
   * @brief Schedule a jobs via the Eigen ThreadPool.  This call simply wraps
   *        the supplied thread functions and maintains internal counters and
   *        tracking structures as the jobs run
   *
   * @param thread_fn    The thread function to run for the job
   * @return std::size_t The Job ID for the job, which can be used to jon a
   *                     specific scheduled job using join()
   */
  std::size_t Schedule(std::function<void()>&& thread_fn);

  /**
   * @brief Join a specific job, given the Job ID
   *
   * @param job_id The ID of the job to join
   */
  void join(std::size_t job_id);
  bool try_join(std::size_t job_id, absl::Duration timeout);

  /**
   * @brief Join all jobs (similar in concept to a FLUSH call)
   *
   * @param allow_reuse If 'false', any subsequent calls to Schedule() will
   *                    cause a runtime error
   */
  void join_all(bool allow_reuse = true);

  /**
   * @brief Join all jobs with timeout
   *
   * @param timeout Amount of time to wait for the jobs to complete
   * @return        true if the jobs were joined within the specified timeout
   *                period
   */
  bool try_join_all(absl::Duration timeout, bool allow_reuse = true);

  /**
   * @brief Join all jobs and forbid any additional jobs to be scheduled
   *        (essentially calls join_all(true))
   *
   */
  void final_join_all();

  /**
   * @brief Check if the thread pool group has a valid thread pool pointer
   *
   * @return true   If this object has a async thread pool
   * @return false  If this object was passed a null async thread pool and
   *                cannot be used
   */
  explicit operator bool() const;

  /**
   * @brief Delegate calls to CurrentThreadId() to the wrapped ThreadPool object
   *
   * @return constexpr std::size_t The current thread ID
   */
  // std::size_t CurrentThreadId() const;

  /**
   * @brief Get the Current Thread Id object from the thread_local ID that we
   * stored at the beginning of the job.  This should be the same as
   * CurrentThreadId() as long as the job was launched from the thread pool.
   *
   * @return std::size_t The current Thread Id
   */
  // static std::size_t GetCurrentThreadLocalId();

  /**
   * @brief Operators for pointer access ( -> ) in order to behave more like an
   *        Eigen::ThreadPool pointer
   */
  constexpr HmThreadPool* operator->() {
    return this;
  }
  constexpr const HmThreadPool* const operator->() const {
    return this;
  }

  std::size_t GetNThreads() const {
    return thread_pool_->NumThreads();
  }

  /**
   * @brief Conveniently create a milliseconds duration for times waits
   *
   * @param milliseconds    Number of milliseconds
   * @return constexpr      absl::Duration representing the given number of
   *                        milliseconds
   */
  static constexpr absl::Duration ms(std::size_t milliseconds) {
    return absl::Milliseconds(milliseconds);
  }

  static Eigen::ThreadPool* get_base_thread_pool();

 private:
  Eigen::ThreadPool* get_internal_thread_pool() const {
    return thread_pool_;
  }

  Eigen::ThreadPool* thread_pool_;

  /** @brief Caller-level mutex blocking parallel calls to Schedule() and
   *         join_all()
   */
  absl::Mutex schedule_mutex_ ABSL_ACQUIRED_BEFORE(mu_);

  /**
   * @brief Mutex prootecting the controller variables
   */
  absl::Mutex mu_ ABSL_ACQUIRED_AFTER(schedule_mutex_);

  std::unordered_set<std::size_t> active_job_ids_ ABSL_GUARDED_BY(mu_){};

  bool final_join_called_ ABSL_GUARDED_BY(schedule_mutex_){false};

  // Thread-local stored pool thread id (not the same as the OS's thread id)
  // static std::size_t thread_local current_thread_pool_thread_id_ /*NOLINT*/;

  static std::atomic<std::size_t> next_job_id_;
};

using ThreadPool = HmThreadPool;

} // namespace hm
