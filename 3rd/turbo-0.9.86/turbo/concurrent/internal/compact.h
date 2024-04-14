
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#ifndef TURBO_CONCURRENT_INTERNAL_COMPACT_H_
#define TURBO_CONCURRENT_INTERNAL_COMPACT_H_

#include "turbo/platform/port.h"
#include <pthread.h>

#if defined(TURBO_PLATFORM_OSX)

#include <sys/cdefs.h>
#include <stdint.h>
#include <dispatch/dispatch.h>    // dispatch_semaphore
#include <errno.h>                // EINVAL

__BEGIN_DECLS

// Implement pthread_spinlock_t for MAC.
struct pthread_spinlock_t {
    dispatch_semaphore_t sem;
};
inline int pthread_spin_init(pthread_spinlock_t *__lock, int __pshared) {
    if (__pshared != 0) {
        return EINVAL;
    }
    __lock->sem = dispatch_semaphore_create(1);
    return 0;
}
inline int pthread_spin_destroy(pthread_spinlock_t *__lock) {
    // TODO(gejun): Not see any destructive API on dispatch_semaphore
    (void) __lock;
    return 0;
}
inline int pthread_spin_lock(pthread_spinlock_t *__lock) {
    return (int) dispatch_semaphore_wait(__lock->sem, DISPATCH_TIME_FOREVER);
}
inline int pthread_spin_trylock(pthread_spinlock_t *__lock) {
    if (dispatch_semaphore_wait(__lock->sem, DISPATCH_TIME_NOW) == 0) {
        return 0;
    }
    return EBUSY;
}
inline int pthread_spin_unlock(pthread_spinlock_t *__lock) {
    return dispatch_semaphore_signal(__lock->sem);
}

__END_DECLS

#elif defined(TURBO_PLATFORM_LINUX)

#include <sys/epoll.h>

#else

#error "The platform does not support epoll-like APIs"

#endif // defined(TURBO_PLATFORM_OSX)

__BEGIN_DECLS

inline uint64_t pthread_numeric_id() {
#if defined(TURBO_PLATFORM_OSX)
    uint64_t id;
    if (pthread_threadid_np(pthread_self(), &id) == 0) {
        return id;
    }
    return -1;
#else
    return pthread_self();
#endif
}

__END_DECLS

#endif  // TURBO_CONCURRENT_INTERNAL_COMPACT_H_
