// Copyright 2023 The titan-search Authors.
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
#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>

namespace clog::details {

    namespace {
        inline uint32_t lossy_increment(std::atomic<uint32_t> *counter) {
            const uint32_t value = counter->load(std::memory_order_relaxed);
            counter->store(value + 1, std::memory_order_relaxed);
            return value;
        }

    }  // namespace

    // Stateful condition class name should be "Log" + name + "State".
    class LogEveryNState final {
    public:
        bool should_log(int n) {
            return n > 0 && (lossy_increment(&counter_) % static_cast<uint32_t>(n)) == 0;
        }

        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    class LogFirstNState final {
    public:
        bool should_log(int n) {
            const uint32_t counter_value = counter_.load(std::memory_order_relaxed);
            if (static_cast<int64_t>(counter_value) < n) {
                counter_.store(counter_value + 1, std::memory_order_relaxed);
                return true;
            }
            return false;
        }

        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    class LogEveryPow2State final {
    public:
        bool should_log() {
            const uint32_t new_value = lossy_increment(&counter_) + 1;
            return (new_value & (new_value - 1)) == 0;
        }

        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    template<size_t Nanos>
    class LogEveryNDurationState final {
    public:
        static constexpr std::chrono::nanoseconds LOG_TIME_PERIOD = std::chrono::nanoseconds(Nanos);

        bool should_log() {
            static std::atomic<int64_t> LogPreviousTimeRaw{0};
            const auto log_current_time =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch());

            const auto LOG_PREVIOUS_TIME =
                    _last_log_cycle.load(std::memory_order_relaxed);
            const auto LOG_TIME_DELTA =
                    log_current_time - std::chrono::nanoseconds(LOG_PREVIOUS_TIME);
            if (LOG_TIME_DELTA > LOG_TIME_PERIOD)
                _last_log_cycle.store(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(log_current_time).count(),std::memory_order_relaxed);
            if (LOG_TIME_DELTA > LOG_TIME_PERIOD) {
                return true;
            }
            return false;
        }

    private:
        // Cycle count according to CycleClock that we should next log at.
        std::atomic<int64_t> _last_log_cycle{0};
    };

}  // namespace clog::details