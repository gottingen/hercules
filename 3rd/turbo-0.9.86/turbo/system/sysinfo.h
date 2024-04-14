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

#ifndef TURBO_STSTEM_SYSINFO_H_
#define TURBO_STSTEM_SYSINFO_H_

#include "turbo/platform/port.h"
#include "turbo/system/internal/sysinfo.h"
#include <cstdio>

namespace turbo {

    /**
     * @ingroup turbo_base
     * @brief Get the current process id
     * @return The current process id
     */
    TURBO_DLL int pid() noexcept;

    /**
     * @ingroup turbo_base
     * @brief Determine if the terminal supports colors
     *        Source: https://github.com/agauniyal/rang/
     * @return
     */
    TURBO_DLL bool is_color_terminal() noexcept;

    /**
     * @ingroup turbo_base
     * @brief Determine if the terminal attached
     *        Source: https://github.com/agauniyal/rang/
     *        @see is_color_terminal()
     * @param file The file to check
     * @return True if the terminal is attached, false otherwise
     */
    TURBO_DLL bool in_terminal(std::FILE *file) noexcept;

    /**
     * @ingroup turbo_base
     * @brief Get the current thread id, this is thread local, so it is unique per thread.
     *        different from std::this_thread::get_id(). actually it is act as gettid() in linux.
     * @return The current thread id
     */
    TURBO_DLL size_t thread_id() noexcept;

    /**
     * @ingroup turbo_base
     * @brief Get the current thread id, on linux same as pthread_self().
     */
    uint64_t thread_numeric_id();

    /**
     * @ingroup turbo_base
     * @brief Get the nominal cpu frequency of the current machine.
     *        Nominal core processor cycles per second of each processor.   This is _not_
     *        necessarily the frequency of the CycleClock counter
     *        Thread-safe.
     * @return
     */
   inline double nominal_cpu_frequency() {
        return base_internal::NominalCPUFrequency();
    }

    /**
     * @ingroup turbo_base
     * @brief Get the number of logical processors (hyperthreads) in system.
     *        Thread-safe.
     * @return
     */
    inline size_t cpu_num() {
        return base_internal::NumCPUs();
    }

}  // namespace turbo

#endif  // TURBO_STSTEM_SYSINFO_H_
