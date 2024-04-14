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


#ifndef TURBO_HASH_INTERNAL_MIXING_HASH_STATE_H_
#define TURBO_HASH_INTERNAL_MIXING_HASH_STATE_H_

#include "turbo/base/bits.h"
#include "turbo/platform/port.h"
#include "turbo/hash/internal/hash_state_base.h"
#include "turbo/hash/hash_engine.h"
#include "turbo/base/endian.h"
#include "turbo/hash/fwd.h"

namespace turbo {
    class HashState;
} // namespace turbo

namespace turbo::hash_internal {

    // MixingHashState
    template<typename Tag>
    class TURBO_DLL MixingHashState : public HashStateBase<MixingHashState<Tag>> {
        // turbo::uint128 is not an alias or a thin wrapper around the intrinsic.
        // We use the intrinsic when available to improve performance.
#ifdef TURBO_HAVE_INTRINSIC_INT128
        using uint128 = __uint128_t;
#else   // TURBO_HAVE_INTRINSIC_INT128
        using uint128 = turbo::uint128;
#endif  // TURBO_HAVE_INTRINSIC_INT128

        template<typename T>
        using IntegralFastPath =
                std::conjunction<std::is_integral<T>, is_uniquely_represented<T>>;

    public:
        // Move only
        MixingHashState(MixingHashState &&) = default;

        MixingHashState &operator=(MixingHashState &&) = default;

        // MixingHashState::combine_contiguous()
        //
        // Fundamental base case for hash recursion: mixes the given range of bytes
        // into the hash state.
        static MixingHashState combine_contiguous(MixingHashState hash_state,
                                                  const unsigned char *first,
                                                  size_t size) {
            return MixingHashState(
                    combine_contiguous_impl(hash_state.state_, first, size,
                                          std::integral_constant<int, sizeof(size_t)>{}));
        }

        using MixingHashState::HashStateBase::combine_contiguous;

        // MixingHashState::hash()
        //
        // For performance reasons in non-opt mode, we specialize this for
        // integral types.
        // Otherwise we would be instantiating and calling dozens of functions for
        // something that is just one multiplication and a couple xor's.
        // The result should be the same as running the whole algorithm, but faster.
        template<typename T, std::enable_if_t<IntegralFastPath<T>::value, int> = 0>
        static size_t hash(T value) {
            return hasher_engine<Tag>::mix(static_cast<uint64_t>(value));
        }

        // Overload of MixingHashState::hash()
        template<typename T, std::enable_if_t<!IntegralFastPath<T>::value, int> = 0>
        static size_t hash(const T &value) {
            return static_cast<size_t>( HashStateBase<MixingHashState<Tag>>::combine(MixingHashState{}, value).state_);
        }

    private:
        // Invoked only once for a given argument; that plus the fact that this is
        // move-only ensures that there is only one non-moved-from object.
        MixingHashState() : state_(hasher_engine<Tag>::Seed()) {}

        friend class MixingHashState::HashStateBase;

        template<typename CombinerT>
        static MixingHashState run_combine_unordered(MixingHashState state,
                                                   CombinerT combiner) {
            uint64_t unordered_state = 0;
            combiner(MixingHashState{}, [&](MixingHashState &inner_state) {
                // Add the hash state of the element to the running total, but mix the
                // carry bit back into the low bit.  This in intended to avoid losing
                // entropy to overflow, especially when unordered_multisets contain
                // multiple copies of the same value.
                auto element_state = inner_state.state_;
                unordered_state += element_state;
                if (unordered_state < element_state) {
                    ++unordered_state;
                }
                inner_state = MixingHashState{};
            });
            return  HashStateBase<MixingHashState<Tag>>::combine(std::move(state), unordered_state);
        }

        // Allow the HashState type-erasure implementation to invoke
        // RunCombinedUnordered() directly.
        friend class turbo::HashState;

        // Workaround for MSVC bug.
        // We make the type copyable to fix the calling convention, even though we
        // never actually copy it. Keep it private to not affect the public API of the
        // type.
        MixingHashState(const MixingHashState &) = default;

        explicit MixingHashState(uint64_t state) : state_(state) {}

        // Implementation of the base case for combine_contiguous where we actually
        // mix the bytes into the state.
        // Dispatch to different implementations of the combine_contiguous depending
        // on the value of `sizeof(size_t)`.
        static uint64_t combine_contiguous_impl(uint64_t state,
                                              const unsigned char *first, size_t len,
                                              std::integral_constant<int, 4>
                                              /* sizeof_size_t */);

        static uint64_t combine_contiguous_impl(uint64_t state,
                                              const unsigned char *first, size_t len,
                                              std::integral_constant<int, 8>
                                              /* sizeof_size_t */);

        // Slow dispatch path for calls to combine_contiguous_impl with a size argument
        // larger than PiecewiseChunkSize().  Has the same effect as calling
        // combine_contiguous_impl() repeatedly with the chunk stride size.
        static uint64_t CombineLargeContiguousImpl32(uint64_t state,
                                                     const unsigned char *first,
                                                     size_t len);

        static uint64_t CombineLargeContiguousImpl64(uint64_t state,
                                                     const unsigned char *first,
                                                     size_t len);

        uint64_t state_;
    };

    // MixingHashState::combine_contiguous_impl()
    template<typename Tag>
    inline uint64_t MixingHashState<Tag>::combine_contiguous_impl(
            uint64_t state, const unsigned char *first, size_t len,
            std::integral_constant<int, 4> /* sizeof_size_t */) {
        // For large values we use LowLevelHash or CityHash depending on the platform,
        // for small ones we just use a multiplicative hash.
        if (TURBO_UNLIKELY(len > PiecewiseChunkSize())) {
            return CombineLargeContiguousImpl32(state, first, len);
        }
        if(TURBO_UNLIKELY(len > 0)) {
            return hasher_engine<Tag>::hash32_with_seed(reinterpret_cast<const char *>(first), len, state);
        }

        return state;
    }

    // Overload of MixingHashState::combine_contiguous_impl()
    template<typename Tag>
    inline uint64_t MixingHashState<Tag>::combine_contiguous_impl(
            uint64_t state, const unsigned char *first, size_t len,
            std::integral_constant<int, 8> /* sizeof_size_t */) {
        // For large values we use LowLevelHash or CityHash depending on the platform,
        // for small ones we just use a multiplicative hash.
        if (TURBO_UNLIKELY(len > PiecewiseChunkSize())) {
            return CombineLargeContiguousImpl64(state, first, len);
        }
        if(TURBO_UNLIKELY(len > 0)) {
            return  hasher_engine<Tag>::hash64_with_seed(reinterpret_cast<const char *>(first), len, state);
        }
        return state;
    }

    extern template class MixingHashState<bytes_hash_tag>;
    extern template class MixingHashState<m3_hash_tag>;
    extern template class MixingHashState<xx_hash_tag>;
}  // namespace turbo::hash_internal

#endif  // TURBO_HASH_INTERNAL_MIXING_HASH_STATE_H_
