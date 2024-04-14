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


#ifndef TURBO_TLOG_CONDITION_H_
#define TURBO_TLOG_CONDITION_H_

#include <atomic>
#include <cstdint>

namespace turbo::tlog::details {

    // Stateful condition class name should be "Log" + name + "State".
    class LogEveryNState final {
    public:
        bool ShouldLog(int n);
        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    class LogFirstNState final {
    public:
        bool ShouldLog(int n);
        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    class LogEveryPow2State final {
    public:
        bool ShouldLog();
        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
    };

    class LogEveryNSecState final {
    public:
        bool ShouldLog(double seconds);
        uint32_t counter() { return counter_.load(std::memory_order_relaxed); }

    private:
        std::atomic<uint32_t> counter_{0};
        // Cycle count according to CycleClock that we should next log at.
        std::atomic<int64_t> next_log_time_cycles_{0};
    };
}  // namespace turbo::tlog::details

#endif  // TURBO_TLOG_CONDITION_H_
