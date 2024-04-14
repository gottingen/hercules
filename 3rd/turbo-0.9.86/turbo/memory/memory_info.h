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


#ifndef TURBO_MEMORY_MEMORY_INFO_H_
#define TURBO_MEMORY_MEMORY_INFO_H_


#include "turbo/platform/port.h"

namespace turbo {

    /**
     * @ingroup turbo_memory_info
     * @brief Get the system memory size.
     * @return The system memory size
     */
    size_t get_system_memory();

    /**
     * @ingroup turbo_memory_info
     * @brief Get the total memory used.
     * @return The total memory used
     */
    size_t get_total_memory_used();

    /**
     * @ingroup turbo_memory_info
     * @brief Get the process memory used by the current process.
     * @return The process memory used
     */
    size_t get_process_memory_used();

    /**
     * @ingroup turbo_memory_info
     * @brief Get the physical memory size.
     * @return The physical memory size
     */
    size_t get_physical_memory();

    /**
     * @ingroup turbo_memory_info
     * @brief Get the memory page size.
     * @return The memory page size
     */
    size_t get_page_size();

    size_t get_peak_memory_used();

    size_t get_current_memory_used();

}  // namespace turbo
#endif  // TURBO_MEMORY_MEMORY_INFO_H_
