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

#ifndef TURBO_HASH_MIX_SIMPLE_MIX_H_
#define TURBO_HASH_MIX_SIMPLE_MIX_H_

#include "turbo/hash/fwd.h"
#include <cstdint>
#include "turbo/platform/port.h"

#if TURBO_HAVE_INTRINSIC_INT128

#include "turbo/base/int128.h"

#endif

namespace turbo {

    namespace simple_internal {
        static constexpr uint64_t kMul = 0xcc9e2d51UL;

        static constexpr size_t mix4(size_t a) {
            uint64_t l = a * kMul;
            return static_cast<size_t>(l ^ (l >> 32));
        }

#if TURBO_HAVE_INTRINSIC_INT128
        static constexpr uint64_t kLo = 0xde5fb9d2630458e9ULL;

        static constexpr size_t mix8(size_t a) {
            uint128 l = uint128(a) * kLo;
            return static_cast<size_t>(l.high64() + l.low64());
        }

#else
        static constexpr size_t mix8(size_t a) {
            a = (~a) + (a << 21); // a = (a << 21) - a - 1;
            a = a ^ (a >> 24);
            a = (a + (a << 3)) + (a << 8); // a * 265
            a = a ^ (a >> 14);
            a = (a + (a << 2)) + (a << 4); // a * 21
            a = a ^ (a >> 28);
            a = a + (a << 31);
            return static_cast<size_t>(a);
        }
#endif
    }  // namespace simple_internal
    /**
     * @brief Simple mix for integer hash.
     * @tparam n size of integer
     */

    template<int N>
    struct simple_mix_impl {
        static constexpr size_t mix(size_t key);
    };

    template<>
    struct simple_mix_impl<4> {
        static constexpr size_t mix(size_t key) {
            return simple_internal::mix4(key);;
        }
    };

    template<>
    struct simple_mix_impl<8> {
        static constexpr size_t mix(size_t key) {
            return simple_internal::mix4(key);;
        }
    };

    /**
     * @ingroup turbo_hash_mixer
     * @brief Simple mix for integer hash integer,
     *        and generate 8 bytes or 4 bytes hash value.
     */
    struct simple_mix {
        using mix4  = simple_mix_impl<4>;
        using mix8  =simple_mix_impl<8>;
    };

    template<>
    struct is_mix_engine<simple_mix> : public std::true_type {};
}  // namespace turbo

#endif  // TURBO_HASH_MIX_SIMPLE_MIX_H_
