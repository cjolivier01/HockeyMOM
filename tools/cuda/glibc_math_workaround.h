#ifndef HM_CUDA_GLIBC_MATH_WORKAROUND_H_
#define HM_CUDA_GLIBC_MATH_WORKAROUND_H_

#include <pthread.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

int pthread_cond_clockwait(
    pthread_cond_t* __restrict __cond,
    pthread_mutex_t* __restrict __mutex,
    clockid_t __clockid,
    const struct timespec* __restrict __abstime);

int pthread_mutex_clocklock(
    pthread_mutex_t* __restrict __mutex,
    clockid_t __clockid,
    const struct timespec* __restrict __abstime);

int pthread_rwlock_clockrdlock(
    pthread_rwlock_t* __restrict __rwlock,
    clockid_t __clockid,
    const struct timespec* __restrict __abstime);

int pthread_rwlock_clockwrlock(
    pthread_rwlock_t* __restrict __rwlock,
    clockid_t __clockid,
    const struct timespec* __restrict __abstime);

#ifdef __cplusplus
}
#endif

#endif  // HM_CUDA_GLIBC_MATH_WORKAROUND_H_
