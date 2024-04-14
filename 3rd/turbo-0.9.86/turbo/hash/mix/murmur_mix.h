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


#ifndef TURBO_HASH_MIX_MURMUR_MIX_H_
#define TURBO_HASH_MIX_MURMUR_MIX_H_

#include "turbo/hash/fwd.h"
#include <cstdint>
#include "turbo/platform/port.h"

namespace turbo {

    namespace murmur_internal {

        static constexpr size_t mix4(size_t h) {
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            return h;
        }

        static constexpr size_t mix8(size_t k) {
            k ^= k >> 33;
            k *= 0xff51afd7ed558ccdUL;
            k ^= k >> 33;
            k *= 0xc4ceb9fe1a85ec53UL;
            k ^= k >> 33;
            return k;
        }
    }  // namespace murmur_internal
    /**
     * @brief Simple mix for integer hash.
     * @tparam n size of integer
     */

    template<int N>
    struct murmur_mix_impl {
        static constexpr size_t mix(size_t key);
    };

    template<>
    struct murmur_mix_impl<4> {
        static constexpr size_t mix(size_t key) {
            return murmur_internal::mix4(key);;
        }
    };

    template<>
    struct murmur_mix_impl<8> {
        static constexpr size_t mix(size_t key) {
            return murmur_internal::mix4(key);;
        }
    };

    /**
     * @ingroup turbo_hash_mixer
     * @brief murmur_mix is a mix engine for integer hash.
     */
    struct murmur_mix {
        using mix4  = murmur_mix_impl<4>;
        using mix8  = murmur_mix_impl<8>;
    };

    template<>
    struct is_mix_engine<murmur_mix> : public std::true_type {};
}  // namespace turbo

#endif  // TURBO_HASH_MIX_MURMUR_MIX_H_
