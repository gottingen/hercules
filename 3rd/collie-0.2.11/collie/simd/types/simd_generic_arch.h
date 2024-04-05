// Copyright 2024 The Elastic-AI Authors.
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

#ifndef COLLIE_SIMD_TYPES_GENERIC_ARCH_H_
#define COLLIE_SIMD_TYPES_GENERIC_ARCH_H_

#include <collie/port/simd_config.h>

/**
 * @defgroup architectures Architecture description
 * */
namespace collie::simd {
    /**
     * @ingroup architectures
     *
     * Base class for all architectures.
     */
    struct generic {
        /// Whether this architecture is supported at compile-time.
        static constexpr bool supported() noexcept { return true; }

        /// Whether this architecture is available at run-time.
        static constexpr bool available() noexcept { return true; }

        /// If this architectures supports aligned memory accesses, the required
        /// alignment.
        static constexpr std::size_t alignment() noexcept { return 0; }

        /// Whether this architecture requires aligned memory access.
        static constexpr bool requires_alignment() noexcept { return false; }

        /// Unique identifier for this architecture.
        static constexpr unsigned version() noexcept { return generic::version(0, 0, 0); }

        /// Name of the architecture.
        static constexpr char const *name() noexcept { return "generic"; }

    protected:
        static constexpr unsigned
        version(unsigned major, unsigned minor, unsigned patch, unsigned multiplier = 100u) noexcept {
            return major * multiplier * multiplier + minor * multiplier + patch;
        }
    };
}

#endif  // COLLIE_SIMD_TYPES_GENERIC_ARCH_H_
