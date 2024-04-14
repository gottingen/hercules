// Copyright 2022 The Turbo Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TURBO_PLATFORM_INTERNAL_CYCLECLOCK_CONFIG_H_
#define TURBO_PLATFORM_INTERNAL_CYCLECLOCK_CONFIG_H_

#include <cstdint>

#include "turbo/base/internal/inline_variable.h"
#include "turbo/platform/port.h"
#include "turbo/platform/internal/unscaledcycleclock_config.h"

namespace turbo::base_internal {

#if TURBO_USE_UNSCALED_CYCLECLOCK
#ifdef NDEBUG
#ifdef TURBO_INTERNAL_UNSCALED_CYCLECLOCK_FREQUENCY_IS_CPU_FREQUENCY
    // Not debug mode and the UnscaledCycleClock frequency is the CPU
    // frequency.  Scale the CycleClock to prevent overflow if someone
    // tries to represent the time as cycles since the Unix epoch.
    TURBO_INTERNAL_INLINE_CONSTEXPR(int32_t, kCycleClockShift, 1);
#else
    // Not debug mode and the UnscaledCycleClock isn't operating at the
    // raw CPU frequency. There is no need to do any scaling, so don't
    // needlessly sacrifice precision.
    TURBO_INTERNAL_INLINE_CONSTEXPR(int32_t, kCycleClockShift, 0);
#endif
#else   // NDEBUG
// In debug mode use a different shift to discourage depending on a
// particular shift value.
    TURBO_INTERNAL_INLINE_CONSTEXPR(int32_t, kCycleClockShift, 2);
#endif  // NDEBUG

    TURBO_INTERNAL_INLINE_CONSTEXPR(double, kCycleClockFrequencyScale,
                                    1.0 / (1 << kCycleClockShift));
#endif  //  TURBO_USE_UNSCALED_CYCLECLOC

}  // namespace turbo::base_internal

#endif  // TURBO_PLATFORM_INTERNAL_CYCLECLOCK_CONFIG_H_
