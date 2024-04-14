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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "turbo/testing/test.h"

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <atomic>
#include "turbo/times/clock.h"
#include "turbo/concurrent/spinlock_wait.h"
#include "turbo/base/processor.h"
#include "turbo/format/print.h"
#include "turbo/times/stop_watcher.h"

namespace plain_test {
    volatile bool stop = false;

    std::atomic<int> nthread(0);

    void* read_thread(void* arg) {
        std::atomic<int>* m = (std::atomic<int>*)arg;
        int njob = 0;
        while (!stop) {
            int x;
            while (!stop && (x = *m) != 0) {
                if (x > 0) {
                    while ((x = m->fetch_sub(1)) > 0) {
                        ++njob;
                        const auto start = turbo::time_now();
                        while (turbo::time_now() < start + turbo::Duration::microseconds(10)) {
                        }
                        if (stop) {
                            return new int(njob);
                        }
                    }
                    m->fetch_add(1);
                } else {
                    turbo::cpu_relax();
                }
            }

            ++nthread;
            turbo::concurrent_internal::futex_wait_private(m/*lock1*/, 0/*consumed_njob*/, nullptr);
            --nthread;
        }
        return new int(njob);
    }

    TEST_CASE("FutexTest, rdlock_performance") {
        const size_t N = 100000;
        std::atomic<int> lock1(0);
        pthread_t rth[8];
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(rth); ++i) {
            REQUIRE_EQ(0, pthread_create(&rth[i], nullptr, read_thread, &lock1));
        }

        const auto t1 = turbo::time_now();
        for (size_t i = 0; i < N; ++i) {
            if (nthread) {
                lock1.fetch_add(1);
                turbo::concurrent_internal::futex_wake_private(&lock1, 1);
            } else {
                lock1.fetch_add(1);
                if (nthread) {
                    turbo::concurrent_internal::futex_wake_private(&lock1, 1);
                }
            }
        }
        const auto t2 = turbo::time_now();

        turbo::sleep_for(turbo::Duration::seconds(3));
        stop = true;
        for (int i = 0; i < 10; ++i) {
            turbo::concurrent_internal::futex_wake_private(&lock1, INT_MAX);
            sched_yield();
        }

        int njob = 0;
        int* res;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(rth); ++i) {
            pthread_join(rth[i], (void**)&res);
            njob += *res;
            delete res;
        }

        turbo::println("wake {} times, {}ns each, lock1={} njob={}",
                N, ((t2-t1)/N).to_nanoseconds(), lock1.load(), njob);
        REQUIRE_EQ(N, (size_t)(lock1.load() + njob));
    }

    TEST_CASE("FutexTest, futex_wake_before_wait") {
        int lock1 = 0;
        timespec timeout = { 1, 0 };
        REQUIRE_EQ(0, turbo::concurrent_internal::futex_wake_private(&lock1, INT_MAX));
        REQUIRE_EQ(-1, turbo::concurrent_internal::futex_wait_private(&lock1, 0, &timeout));
        REQUIRE_EQ(ETIMEDOUT, errno);
    }

    void* dummy_waiter(void* lock) {
        turbo::concurrent_internal::futex_wait_private(lock, 0, nullptr);
        return nullptr;
    }

    TEST_CASE("FutexTest, futex_wake_many_waiters_perf") {

        int lock1 = 0;
        size_t N = 0;
        pthread_t th;
        for (; N < 1000 && !pthread_create(&th, nullptr, dummy_waiter, &lock1); ++N) {}

        sleep(1);
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            nwakeup += turbo::concurrent_internal::futex_wake_private(&lock1, 1);
        }
        tm.stop();
        turbo::println("N={}, futex_wake a thread = {}ns", N, tm.elapsed_nano() / N);
        REQUIRE_EQ(N, (size_t)nwakeup);

        sleep(2);
        const size_t REP = 10000;
        nwakeup = 0;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            nwakeup += turbo::concurrent_internal::futex_wake_private(&lock1, 1);
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("N={}, futex_wake nop = {}ns", N, tm.elapsed_nano() / REP);
    }

    std::atomic<int> nevent(0);

    void* waker(void* lock) {
        turbo::sleep_for(turbo::Duration::microseconds(10000));
        const size_t REP = 100000;
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            nwakeup += turbo::concurrent_internal::futex_wake_private(lock, 1);
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("futex_wake nop = {}ns", tm.elapsed_nano() / REP);
        return nullptr;
    }

    void* batch_waker(void* lock) {
        turbo::sleep_for(turbo::Duration::microseconds(10000));
        const size_t REP = 100000;
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            if (nevent.fetch_add(1, std::memory_order_relaxed) == 0) {
                nwakeup += turbo::concurrent_internal::futex_wake_private(lock, 1);
                int expected = 1;
                while (1) {
                    int last_expected = expected;
                    if (nevent.compare_exchange_strong(expected, 0, std::memory_order_relaxed)) {
                        break;
                    }
                    nwakeup += turbo::concurrent_internal::futex_wake_private(lock, expected - last_expected);
                }
            }
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("futex_wake nop = {}ns", tm.elapsed_nano() / REP);
        return nullptr;
    }

    TEST_CASE("FutexTest, many_futex_wake_nop_perf") {
        pthread_t th[8];
        int lock1;
        std::cout << "[Direct wake]" << std::endl;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_create(&th[i], nullptr, waker, &lock1));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_join(th[i], nullptr));
        }
        std::cout << "[Batch wake]" << std::endl;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_create(&th[i], nullptr, batch_waker, &lock1));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_join(th[i], nullptr));
        }
    }
} // namespace plain_test

namespace spin_waiter_test {
    volatile bool stop_waiter = false;

    std::atomic<int> nthread(0);

    void* read_thread(void* arg) {
        auto m = (turbo::SpinWaiter*)arg;
        int njob = 0;
        while (!stop_waiter) {
            int x;
            while (!stop_waiter && (x = *m) != 0) {
                if (x > 0) {
                    while ((x = m->fetch_sub(1)) > 0) {
                        ++njob;
                        const auto start = turbo::time_now();
                        while (turbo::time_now() < start + turbo::Duration::microseconds(10)) {
                        }
                        if (stop_waiter) {
                            return new int(njob);
                        }
                    }
                    m->fetch_add(1);
                } else {
                    turbo::cpu_relax();
                }
            }

            ++nthread;
            m->wait();
            --nthread;
        }
        return new int(njob);
    }

    TEST_CASE("FutexTest, rdlock_performance") {
        const size_t N = 100000;
        turbo::SpinWaiter lock1(0);
        pthread_t rth[8];
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(rth); ++i) {
            REQUIRE_EQ(0, pthread_create(&rth[i], nullptr, read_thread, &lock1));
        }

        const auto t1 = turbo::time_now();
        for (size_t i = 0; i < N; ++i) {
            if (nthread) {
                lock1.fetch_add(1);
               lock1.wake_one();
            } else {
                lock1.fetch_add(1);
                if (nthread) {
                    lock1.wake_one();
                }
            }
        }
        const auto t2 = turbo::time_now();

        turbo::sleep_for(turbo::Duration::seconds(3));
        stop_waiter = true;
        for (int i = 0; i < 10; ++i) {
            lock1.wake_all();
            sched_yield();
        }

        int njob = 0;
        int* res;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(rth); ++i) {
            pthread_join(rth[i], (void**)&res);
            njob += *res;
            delete res;
        }

        turbo::println("wake {} times, {}ns each, lock1={} njob={}",
                       N, ((t2-t1)/N).to_nanoseconds(), lock1.load(), njob);
        REQUIRE_EQ(N, (size_t)(lock1.load() + njob));
    }

    TEST_CASE("FutexTest, futex_wake_before_wait") {
        turbo::SpinWaiter lock1;
        lock1.wake_all();
        auto rs = lock1.wait_for(0, turbo::Duration::seconds(1));
        REQUIRE(turbo::is_deadline_exceeded(rs));
    }

    void* dummy_spin_waiter(void* lock_ptr) {
        auto lock = (turbo::SpinWaiter*)lock_ptr;
        lock->wait();
        return nullptr;
    }

    TEST_CASE("FutexTest, futex_wake_many_waiters_perf") {

        turbo::SpinWaiter lock1(0);
        size_t N = 0;
        pthread_t th;
        for (; N < 1000 && !pthread_create(&th, nullptr, dummy_spin_waiter, &lock1); ++N) {}

        sleep(1);
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < N; ++i) {
            nwakeup += lock1.wake_one();
        }
        tm.stop();
        turbo::println("N={}, futex_wake a thread = {}ns", N, tm.elapsed_nano() / N);
        REQUIRE_EQ(N, (size_t)nwakeup);

        sleep(2);
        const size_t REP = 10000;
        nwakeup = 0;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            nwakeup += lock1.wake_one();
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("N={}, futex_wake nop = {}ns", N, tm.elapsed_nano() / REP);
    }

    std::atomic<int> nevent(0);

    void* waker(void* lock_ptr) {
        turbo::sleep_for(turbo::Duration::microseconds(10000));
        auto lock = (turbo::SpinWaiter*)lock_ptr;
        const size_t REP = 100000;
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            lock->wake_one();
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("futex_wake nop = {}ns", tm.elapsed_nano() / REP);
        return nullptr;
    }

    void* batch_waker(void* lock_ptr) {
        turbo::sleep_for(turbo::Duration::microseconds(10000));
        auto lock = (turbo::SpinWaiter*)lock_ptr;
        const size_t REP = 100000;
        int nwakeup = 0;
        turbo::StopWatcher tm;
        tm.reset();
        for (size_t i = 0; i < REP; ++i) {
            if (nevent.fetch_add(1, std::memory_order_relaxed) == 0) {
                lock->wake_one();
                int expected = 1;
                while (1) {
                    int last_expected = expected;
                    if (nevent.compare_exchange_strong(expected, 0, std::memory_order_relaxed)) {
                        break;
                    }
                    nwakeup += lock->wake(expected - last_expected);
                }
            }
        }
        tm.stop();
        REQUIRE_EQ(0, nwakeup);
        turbo::println("futex_wake nop = {}ns", tm.elapsed_nano() / REP);
        return nullptr;
    }

    TEST_CASE("FutexTest, many_futex_wake_nop_perf") {
        pthread_t th[8];
        int lock1;
        std::cout << "[Direct wake]" << std::endl;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_create(&th[i], nullptr, waker, &lock1));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_join(th[i], nullptr));
        }
        std::cout << "[Batch wake]" << std::endl;
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_create(&th[i], nullptr, batch_waker, &lock1));
        }
        for (size_t i = 0; i < TURBO_ARRAY_SIZE(th); ++i) {
            REQUIRE_EQ(0, pthread_join(th[i], nullptr));
        }
    }

}  // namespace spin_waiter_test
