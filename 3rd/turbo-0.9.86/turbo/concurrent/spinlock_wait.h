// Copyright 2020 The Turbo Authors.
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

#ifndef TURBO_BASE_INTERNAL_SPINLOCK_WAIT_H_
#define TURBO_BASE_INTERNAL_SPINLOCK_WAIT_H_

// Operations to make atomic transitions on a word, and to allow
// waiting for those transitions to become possible.

#include <stdint.h>
#include <atomic>
#include "turbo/times/clock.h"
#include "turbo/platform/internal/scheduling_mode.h"
#include "turbo/concurrent/internal/futex.h"
#include "turbo/status/status.h"

namespace turbo {

    // spin_lock_wait() waits until it can perform one of several transitions from
    // "from" to "to".  It returns when it performs a transition where done==true.
    struct SpinLockWaitTransition {
        uint32_t from;
        uint32_t to;
        bool done;
    };

    // Wait until *w can transition from trans[i].from to trans[i].to for some i
    // satisfying 0<=i<n && trans[i].done, atomically make the transition,
    // then return the old value of *w.   Make any other atomic transitions
    // where !trans[i].done, but continue waiting.
    //
    // Wakeups for threads blocked on spin_lock_wait do not respect priorities.
    uint32_t spin_lock_wait(std::atomic<uint32_t> *w, int n,
                          const SpinLockWaitTransition trans[],
                          base_internal::SchedulingMode scheduling_mode);

    // If possible, wake some thread that has called spin_lock_delay(w, ...). If `all`
    // is true, wake all such threads. On some systems, this may be a no-op; on
    // those systems, threads calling spin_lock_delay() will always wake eventually
    // even if spin_lock_wake() is never called.
    void spin_lock_wake(std::atomic<uint32_t> *w, bool all);

    // Wait for an appropriate spin delay on iteration "loop" of a
    // spin loop on location *w, whose previously observed value was "value".
    // spin_lock_delay() may do nothing, may yield the CPU, may sleep a clock tick,
    // or may wait for a call to spin_lock_wake(w).
    void spin_lock_delay(std::atomic<uint32_t> *w, uint32_t value, int loop,
                       base_internal::SchedulingMode scheduling_mode);

    // Helper used by turbo_internal_spin_lock_delay.
    // Returns a suggested delay in nanoseconds for iteration number "loop".
    int spin_lock_suggested_delay_ns(int loop);

    class SpinWaiter {
    public:
        SpinWaiter()  = default;

        explicit SpinWaiter(int n){
            _waiter.store(n, std::memory_order_relaxed);
        }

        SpinWaiter(const SpinWaiter&) = delete;
        SpinWaiter& operator=(const SpinWaiter&) = delete;

        SpinWaiter(SpinWaiter&&) = delete;
        SpinWaiter& operator=(SpinWaiter&&) = delete;

        ~SpinWaiter() = default;

        void store(int32_t waiter, std::memory_order order = std::memory_order_relaxed) {
            _waiter.store(waiter, order);
        }

        int32_t load(std::memory_order order = std::memory_order_relaxed) const {
            return _waiter.load(order);
        }

        int32_t operator++() {
            return _waiter.fetch_add(1, std::memory_order_relaxed) + 1;
        }

        int32_t operator--() {
            return _waiter.fetch_sub(1, std::memory_order_relaxed) -1;
        }

        int32_t operator++(int) {
            return _waiter.fetch_add(1, std::memory_order_relaxed);
        }

        int32_t operator--(int) {
            return _waiter.fetch_sub(1, std::memory_order_relaxed);
        }

        int32_t fetch_add(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.fetch_add(n, order);
        }

        int32_t fetch_sub(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.fetch_sub(n, order);
        }

        int32_t fetch_or(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.fetch_or(n, order);
        }

        int32_t fetch_and(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.fetch_and(n, order);
        }

        int32_t fetch_xor(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.fetch_xor(n, order);
        }

        int32_t exchange(int32_t n, std::memory_order order = std::memory_order_relaxed) {
            return _waiter.exchange(n, order);
        }


        SpinWaiter& operator+=(int32_t n) {
            _waiter.fetch_add(n, std::memory_order_relaxed);
            return *this;
        }

        SpinWaiter& operator-=(int32_t n) {
            _waiter.fetch_sub(n, std::memory_order_relaxed);
            return *this;
        }

        SpinWaiter& operator=(int32_t n) {
            _waiter.store(n, std::memory_order_relaxed);
            return *this;
        }

        bool operator==(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) == n;
        }

        bool operator!=(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) != n;
        }

        bool operator<(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) < n;
        }

        bool operator<=(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) <= n;
        }

        bool operator>(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) > n;
        }

        bool operator>=(int32_t n) const {
            return _waiter.load(std::memory_order_relaxed) >= n;
        }

        operator int32_t() const {
            return _waiter.load(std::memory_order_relaxed);
        }

        bool is_zero() const {
            return _waiter.load(std::memory_order_relaxed) == 0;
        }

        void wait(int expected = 0) {
            turbo::concurrent_internal::futex_wait_private(&_waiter, expected, nullptr);
        }

        turbo::Status wait_until(int expected, const turbo::Time& deadline) {
            if(deadline <= time_now()) {
                return deadline_exceeded_error("duration is negative");
            }
            auto spec = (deadline - time_now()).to_timespec();
            auto ret = turbo::concurrent_internal::futex_wait_private(&_waiter, expected, &spec);
            if(ret != 0) {
                return deadline_exceeded_error("wait_until timeout");
            }
            return turbo::ok_status();
        }

        turbo::Status wait_for(int expected, const turbo::Duration& duration) {
            if(duration <= turbo::Duration::zero()) {
                return deadline_exceeded_error("duration is negative");
            }
            auto spec = duration.to_timespec();
            auto ret =turbo::concurrent_internal::futex_wait_private(&_waiter, expected, &spec);
            if(ret != 0) {
                return deadline_exceeded_error("wait_for timeout");
            }
            return turbo::ok_status();
        }

        int wake(int n) {
            return turbo::concurrent_internal::futex_wake_private(&_waiter, n);
        }

        int wake_one() {
            return turbo::concurrent_internal::futex_wake_private(&_waiter, 1);
        }

        int wake_all() {
            return turbo::concurrent_internal::futex_wake_private(&_waiter, INT_MAX);
        }

    private:
        std::atomic<int32_t> _waiter{0};
    };

}  // namespace turbo

// In some build configurations we pass --detect-odr-violations to the
// gold linker.  This causes it to flag weak symbol overrides as ODR
// violations.  Because ODR only applies to C++ and not C,
// --detect-odr-violations ignores symbols not mangled with C++ names.
// By changing our extension points to be extern "C", we dodge this
// check.
extern "C" {
void turbo_internal_spin_lock_wake(std::atomic<uint32_t> *w,
                                                        bool all);
void turbo_internal_spin_lock_delay(
        std::atomic<uint32_t> *w, uint32_t value, int loop,
        turbo::base_internal::SchedulingMode scheduling_mode);
}

inline void turbo::spin_lock_wake(std::atomic<uint32_t> *w,
                                bool all) {
    turbo_internal_spin_lock_wake(w, all);
}

inline void turbo::spin_lock_delay(
        std::atomic<uint32_t> *w, uint32_t value, int loop,
        turbo::base_internal::SchedulingMode scheduling_mode) {
    turbo_internal_spin_lock_delay
            (w, value, loop, scheduling_mode);
}

#endif  // TURBO_BASE_INTERNAL_SPINLOCK_WAIT_H_
