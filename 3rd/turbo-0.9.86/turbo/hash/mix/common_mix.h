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
//
// Created by jeff on 23-12-15.
//

#ifndef TURBO_HASH_MIX_COMMON_MIX_H_
#define TURBO_HASH_MIX_COMMON_MIX_H_

#include "turbo/platform/port.h"

#ifdef TURBO_HAVE_INTRINSIC_INT128
#include "turbo/base/int128.h"
#endif

namespace turbo::hash_internal {


    static constexpr uint64_t kMul =
            sizeof(size_t) == 4 ? uint64_t{0xcc9e2d51}
                                : uint64_t{0x9ddfea08eb382d69};

    inline uint64_t common_mix_with_seed(uint64_t seed, uint64_t value) {
        // Though the 128-bit product on AArch64 needs two instructions, it is
        // still a good balance between speed and hash quality.
        using MultType =
                std::conditional_t<sizeof(size_t) == 4, uint64_t, uint128>;
        // We do the addition in 64-bit space to make sure the 128-bit
        // multiplication is fast. If we were to do it as MultType the compiler has
        // to assume that the high word is non-zero and needs to perform 2
        // multiplications instead of one.
        MultType m = seed + value;
        m *= kMul;
        return static_cast<uint64_t>(m ^ (m >> (sizeof(m) * 8 / 2)));
    }

    inline uint64_t common_mix(uint64_t value) {
        static constexpr uint64_t kSeed = 0xcc9e2d51UL;
        return common_mix_with_seed(kSeed, value);
    }

}  // namespace turbo::hash_internal
#endif  // TURBO_HASH_MIX_COMMON_MIX_H_
