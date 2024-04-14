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
#ifndef TURBO_CONCURRENT_LATCH_H_
#define TURBO_CONCURRENT_LATCH_H_

#include <condition_variable>
#include <mutex>
#include <atomic>
#include <memory>
#include "turbo/times/time.h"
#include "turbo/log/logging.h"

namespace turbo {

    class Latch {
    public:
        explicit Latch(uint32_t count = 0);

        // Decrement internal counter. If it reaches zero, wake up all waiters.
        void CountDown(uint32_t update = 1);

        void CountUp(uint32_t update = 1);

        // Test if the latch's internal counter has become zero.
        [[nodiscard]] bool TryWait() const noexcept;

        // Wait until `latch`'s internal counter reached zero.
        void Wait() const;

        bool WaitFor(const turbo::Duration &d) {
            std::chrono::microseconds timeout = d.to_chrono_microseconds();
            std::unique_lock lk(_data->mutex);
            TLOG_CHECK_GE(_data->count, 0ul);
            return _data->cond.wait_for(lk, timeout, [this] { return _data->count == 0; });
        }

        bool WaitUntil(const turbo::Time &deadline) {
            auto d = deadline.to_chrono_time();
            std::unique_lock lk(_data->mutex);
            TLOG_CHECK_GE(_data->count, 0ul);
            return _data->cond.wait_until(lk, d, [this] { return _data->count == 0; });
        }

        // Shorthand for `count_down(); wait();`
        void ArriveAndWait(std::ptrdiff_t update = 1);

        bool Arrived() {
            return _data->count == 0;
        }

    private:
        struct inner_data {
            std::mutex mutex;
            std::condition_variable cond;
            std::atomic<uint32_t> count{0};
        };
        std::shared_ptr<inner_data> _data;
    };

}  // namespace turbo
#endif  // TURBO_CONCURRENT_LATCH_H_
