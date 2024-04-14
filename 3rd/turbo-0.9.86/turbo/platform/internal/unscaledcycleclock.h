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
//
// UnscaledCycleClock
//    An UnscaledCycleClock yields the value and frequency of a cycle counter
//    that increments at a rate that is approximately constant.
//    This class is for internal use only, you should consider using CycleClock
//    instead.
//
// Notes:
// The cycle counter frequency is not necessarily the core clock frequency.
// That is, CycleCounter cycles are not necessarily "CPU cycles".
//
// An arbitrary offset may have been added to the counter at power on.
//
// On some platforms, the rate and offset of the counter may differ
// slightly when read from different CPUs of a multiprocessor.  Usually,
// we try to ensure that the operating system adjusts values periodically
// so that values agree approximately.   If you need stronger guarantees,
// consider using alternate interfaces.
//
// The CPU is not required to maintain the ordering of a cycle counter read
// with respect to surrounding instructions.

#ifndef TURBO_PLATFORM_INTERNAL_UNSCALEDCYCLECLOCK_H_
#define TURBO_PLATFORM_INTERNAL_UNSCALEDCYCLECLOCK_H_

#include <cstdint>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include "turbo/platform/port.h"
#include "turbo/platform/internal/unscaledcycleclock_config.h"

#if TURBO_USE_UNSCALED_CYCLECLOCK

namespace turbo::time_internal {
    class UnscaledCycleClockWrapperForGetCurrentTime;
}  // namespace turbo::time_internal

namespace turbo::base_internal {
    class CycleClock;

    class UnscaledCycleClockWrapperForInitializeFrequency;

    class UnscaledCycleClock {
    private:
        UnscaledCycleClock() = delete;

        // Return the value of a cycle counter that counts at a rate that is
        // approximately constant.
        static int64_t time_now();

        // Return the how much UnscaledCycleClock::time_now() increases per second.
        // This is not necessarily the core CPU clock frequency.
        // It may be the nominal value report by the kernel, rather than a measured
        // value.
        static double Frequency();

        // Allowed users
        friend class base_internal::CycleClock;

        friend class time_internal::UnscaledCycleClockWrapperForGetCurrentTime;

        friend class base_internal::UnscaledCycleClockWrapperForInitializeFrequency;
    };

#if defined(__x86_64__)

    inline int64_t UnscaledCycleClock::time_now() {
        uint64_t low, high;
        __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
        return static_cast<int64_t>((high << 32) | low);
    }

#endif

}  // namespace turbo::base_internal

#endif  // TURBO_USE_UNSCALED_CYCLECLOCK

#endif  // TURBO_PLATFORM_INTERNAL_UNSCALEDCYCLECLOCK_H_
