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


#ifndef TURBO_HASH_m3_M3_H_
#define TURBO_HASH_m3_M3_H_

#include <cstdint>
#include <cstdlib>  // for size_t.

#include <utility>

#include "turbo/platform/port.h"
#include "turbo/hash/fwd.h"
#include "turbo/hash/m3/murmurhash3.h"


namespace turbo {

    /**
     * @ingroup turbo_hash_engine
     * @brief m3_hash_tag is a tag for m3_hash.
     */
    struct m3_hash_tag {

        static constexpr const char* name() {
            return "city_hash";
        }

        constexpr static bool available() {
            return true;
        }
    };

    template <>
    struct hasher_engine<m3_hash_tag> {

        static uint64_t mix(uint64_t k);

        static uint64_t mix_with_seed(uint64_t seed, uint64_t value);

        static uint32_t hash32(const char *s, size_t len);

        static uint32_t hash32_with_seed(const char *s, size_t len, uint32_t seed);

        static size_t hash64(const char *s, size_t len);

        static size_t hash64_with_seed(const char *s, size_t len, uint64_t seed);

        TURBO_FORCE_INLINE static uint64_t Seed() {
            return 0;
        }
    };

    inline uint64_t hasher_engine<m3_hash_tag>::mix(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdUL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53UL;
        k ^= k >> 33;
        return k;
    }

    inline uint64_t hasher_engine<m3_hash_tag>::mix_with_seed(uint64_t seed, uint64_t value) {
        int64_t k = seed + value;
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdUL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53UL;
        k ^= k >> 33;
        return k;
    }

}  // namespace turbo

#endif  // TURBO_HASH_m3_M3_H_
