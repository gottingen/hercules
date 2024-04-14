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

#include "turbo/concurrent/internal/futex.h"
#if !defined(TURBO_PLATFORM_LINUX)
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <condition_variable>
#include <turbo/container/flat_hash_map.h>
#include "turbo/log/logging.h"
#include <pthread.h>
#include <unordered_map>


namespace turbo {

    class SimuFutex {
    public:
        SimuFutex() : counts(0), ref(0) {

        }

        ~SimuFutex() {
        }

    public:
        std::mutex lock;
        std::condition_variable cond;
        int32_t counts;
        int32_t ref;
    };

    static std::mutex s_futex_map_mutex;
    static std::once_flag init_futex_map_once;

    static std::unordered_map<void *, SimuFutex> *s_futex_map = nullptr;

    static void InitFutexMap() {
        // Leave memory to process's clean up.
        s_futex_map = new(std::nothrow) std::unordered_map<void *, SimuFutex>();
        if (NULL == s_futex_map) {
            exit(1);
        }
        return;
    }

    int futex_wait_private(void *addr1, int expected, const timespec *timeout) {
        try {
            std::call_once(init_futex_map_once, InitFutexMap);
        } catch (const std::system_error &e) {
            TLOG_CRITICAL("Fail to pthread_once");
            exit(1);
        }
        std::unique_lock mu(s_futex_map_mutex);
        SimuFutex &simu_futex = (*s_futex_map)[addr1];
        ++simu_futex.ref;
        mu.unlock();
        int rc = 0;
        {

            std::unique_lock mu1(simu_futex.lock);
            if (static_cast<std::atomic<int> *>(addr1)->load() == expected) {
                ++simu_futex.counts;
                if (timeout) {
                    auto timeout_c = turbo::to_chrono_time(turbo::from_now(*timeout));
                    auto to = simu_futex.cond.wait_until(mu1, timeout_c);
                    if (to != std::cv_status::no_timeout) {
                        rc = -1;
                    }
                } else {
                    simu_futex.cond.wait(mu1);
                }
                --simu_futex.counts;
            } else {
                errno = EAGAIN;
                rc = -1;
            }
        }

        std::unique_lock mu1(s_futex_map_mutex);
        if (--simu_futex.ref == 0) {
            s_futex_map->erase(addr1);
        }
        mu1.unlock();
        return rc;
    }

    int futex_wake_private(void *addr1, int nwake) {
        try {
            std::call_once(init_futex_map_once, InitFutexMap);
        } catch (const std::system_error &e) {
            TLOG_CRITICAL("Fail to pthread_once");
            exit(1);
        }
        std::unique_lock mu(s_futex_map_mutex);
        auto it = s_futex_map->find(addr1);
        if (it == s_futex_map->end()) {
            mu.unlock();
            return 0;
        }
        SimuFutex &simu_futex = it->second;
        ++simu_futex.ref;
        mu.unlock();

        int nwakedup = 0;
        int rc = 0;
        {
            std::unique_lock mu1(simu_futex.lock);
            nwake = (nwake < simu_futex.counts) ? nwake : simu_futex.counts;
            for (int i = 0; i < nwake; ++i) {
                simu_futex.cond.notify_one();
                ++nwakedup;
            }
        }

        std::unique_lock mu2(s_futex_map_mutex);
        if (--simu_futex.ref == 0) {
            s_futex_map->erase(addr1);
        }
        mu2.unlock();
        return nwakedup;
    }

} // namespace turbo

#endif  // !defined(TURBO_PLATFORM_LINUX)