// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//
//
// Created by jeff on 24-1-3.
//

#ifndef TURBO_CONCURRENT_LOCK_H_
#define TURBO_CONCURRENT_LOCK_H_

#include "turbo/concurrent/spinlock.h"
#include "turbo/log/logging.h"

namespace std {

    template<>
    class TURBO_SCOPED_LOCKABLE unique_lock<turbo::SpinLock> {
        TURBO_NON_COPYABLE(unique_lock);
    public:
        unique_lock() : _lock(NULL), _owns_lock(false) {}
        unique_lock(turbo::SpinLock &lock) : _lock(&lock), _owns_lock(false) {
            lock.lock();
            _owns_lock = true;
        }

        bool owns_lock() const noexcept {
            return _owns_lock;
        }

        void *mutex() const noexcept {
            return _lock;
        }

        unique_lock(turbo::SpinLock &lock, std::defer_lock_t) : _lock(&lock), _owns_lock(false) {}

        unique_lock(turbo::SpinLock &lock, std::try_to_lock_t) : _lock(&lock), _owns_lock(lock.try_lock()) {}

        unique_lock(turbo::SpinLock &lock, std::adopt_lock_t) : _lock(&lock), _owns_lock(true) {}

        ~unique_lock() {
            if (_owns_lock) {
                _lock->unlock();
            }
        }

        void lock() {
            TDLOG_CHECK(!owns_lock());
            _lock->lock();
            _owns_lock = true;
        }

        bool try_lock() {
            TDLOG_CHECK(!owns_lock());
            _owns_lock = _lock->try_lock();
            return _owns_lock;
        }

        void unlock() {
            TDLOG_CHECK(owns_lock());
            _lock->unlock();
            _owns_lock = false;
        }
    private:
        turbo::SpinLock *_lock;
        bool _owns_lock;
    };

}  // namespace std

namespace turbo {
    template<typename Mutex1, typename Mutex2>
    void double_lock(std::unique_lock<Mutex1> &lck1, std::unique_lock<Mutex2> &lck2) {
        TDLOG_CHECK(!lck1.owns_lock());
        TDLOG_CHECK(!lck2.owns_lock());
        volatile void *const ptr1 = lck1.mutex();
        volatile void *const ptr2 = lck2.mutex();
        TDLOG_CHECK_NE(ptr1, ptr2);
        if (ptr1 < ptr2) {
            lck1.lock();
            lck2.lock();
        } else {
            lck2.lock();
            lck1.lock();
        }
    }
}
#endif  // TURBO_CONCURRENT_LOCK_H_
