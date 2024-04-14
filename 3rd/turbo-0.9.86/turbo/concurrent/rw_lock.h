// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TURBO_CONCURRENT_RW_LOCK_H_
#define TURBO_CONCURRENT_RW_LOCK_H_

#include <pthread.h>
#include <stdint.h>
#include <mutex>
#include "turbo/platform/port.h"
#include "turbo/concurrent/scoped_lock.h"

namespace turbo {

    enum LockMode {
        INVALID_LOCK, READ_LOCK, WRITE_LOCK
    };

    class RWLock {
    public:
        RWLock() { pthread_rwlock_init(&_lock, nullptr); }

        bool lock(LockMode mode) {
            switch (mode) {
                case READ_LOCK: {
                    return 0 == pthread_rwlock_rdlock(&_lock);
                }
                case WRITE_LOCK: {
                    return 0 == pthread_rwlock_wrlock(&_lock);
                }
                default: {
                    return false;
                }
            }
        }

        bool unlock(LockMode mode) {
            TURBO_UNUSED(mode);
            return 0 == pthread_rwlock_unlock(&_lock);
        }

        ~RWLock() { pthread_rwlock_destroy(&_lock); }

    private:
        pthread_rwlock_t _lock;
    };

    // read only
    class ReadLock {
    public:
        explicit ReadLock(RWLock &lock) : _lock(lock) {}

        void lock() { _lock.lock(READ_LOCK); }

        void unlock() { _lock.unlock(READ_LOCK); }

    private:
        RWLock &_lock;
    };

    class ScopedReadLock {
    public:
        explicit ScopedReadLock(RWLock &lock) : _lock(lock) {
            _lock.lock(READ_LOCK);
        }
        ~ScopedReadLock() {
            _lock.unlock(READ_LOCK);
        }
    private:
        RWLock &_lock;
    };

    // write lock
    class WriteLock {
    public:
        explicit WriteLock(RWLock &lock) : _lock(lock) {
        }

        ~WriteLock() {
        }

        void lock() { _lock.lock(WRITE_LOCK); }

        void unlock() { _lock.unlock(WRITE_LOCK); }

    private:
        RWLock &_lock;
    };

    class ScopedWriteLock {
    public:
        explicit ScopedWriteLock(RWLock &lock) : _lock(lock) {
            _lock.lock(WRITE_LOCK);
        }

        ~ScopedWriteLock() {
            _lock.unlock(WRITE_LOCK);
        }

    private:
        RWLock &_lock;
    };

}  // namespace turbo

#endif  // TURBO_CONCURRENT_RW_LOCK_H_
