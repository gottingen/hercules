// Copyright 2018 The Turbo Authors.
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
// -----------------------------------------------------------------------------
// File: hash.h
// -----------------------------------------------------------------------------
//
#ifndef TURBO_HASH_INTERNAL_HASH_H_
#define TURBO_HASH_INTERNAL_HASH_H_

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <forward_list>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <optional>

#include "turbo/base/int128.h"
#include "turbo/container/fixed_array.h"
#include "turbo/meta/type_traits.h"
#include "turbo/meta/utility.h"
#include <variant>
#include "turbo/platform/port.h"
#include "turbo/platform/internal/unaligned_access.h"
#include "turbo/platform/port.h"
#include "turbo/strings/string_view.h"
#include "turbo/hash/internal/hash_state_base.h"
#include "turbo/hash/internal/hash_select.h"
#include "turbo/hash/internal/utility.h"
#include "turbo/hash/internal/hash_value.h"
#include "turbo/hash/internal/mixing_hash_state.h"

namespace turbo::hash_internal {

    template<typename T, typename Tag>
    struct HashImpl {
        size_t operator()(const T &value) const {
            return MixingHashState<Tag>::hash(value);
        }
    };

    template<typename T, typename Tag>
    struct Hash
            : std::conditional_t<is_hashable<T>::value, HashImpl<T, Tag>, PoisonedHash> {
    };

    template<typename H>
    template<typename T, typename... Ts>
    H HashStateBase<H>::combine(H state, const T &value, const Ts &... values) {
        return H::combine(hash_internal::HashSelect::template Apply<T>::Invoke(
                                  std::move(state), value),
                          values...);
    }

    // HashStateBase::combine_contiguous()
    template<typename H>
    template<typename T>
    H HashStateBase<H>::combine_contiguous(H state, const T *data, size_t size) {
        return hash_internal::hash_range_or_bytes(std::move(state), data, size);
    }

    // HashStateBase::combine_unordered()
    template<typename H>
    template<typename I>
    H HashStateBase<H>::combine_unordered(H state, I begin, I end) {
        return H::run_combine_unordered(std::move(state),
                                      CombineUnorderedCallback<I>{begin, end});
    }

    // HashStateBase::PiecewiseCombiner::add_buffer()
    template<typename H>
    H PiecewiseCombiner::add_buffer(H state, const unsigned char *data,
                                    size_t size) {
        if (position_ + size < PiecewiseChunkSize()) {
            // This partial chunk does not fill our existing buffer
            memcpy(buf_ + position_, data, size);
            position_ += size;
            return state;
        }

        // If the buffer is partially filled we need to complete the buffer
        // and hash it.
        if (position_ != 0) {
            const size_t bytes_needed = PiecewiseChunkSize() - position_;
            memcpy(buf_ + position_, data, bytes_needed);
            state = H::combine_contiguous(std::move(state), buf_, PiecewiseChunkSize());
            data += bytes_needed;
            size -= bytes_needed;
        }

        // Hash whatever chunks we can without copying
        while (size >= PiecewiseChunkSize()) {
            state = H::combine_contiguous(std::move(state), data, PiecewiseChunkSize());
            data += PiecewiseChunkSize();
            size -= PiecewiseChunkSize();
        }
        // Fill the buffer with the remainder
        memcpy(buf_, data, size);
        position_ = size;
        return state;
    }

    // HashStateBase::PiecewiseCombiner::finalize()
    template<typename H>
    H PiecewiseCombiner::finalize(H state) {
        // Hash the remainder left in the buffer, which may be empty
        return H::combine_contiguous(std::move(state), buf_, position_);
    }
}  // namespace turbo::hash_internal

#endif  // TURBO_HASH_INTERNAL_HASH_H_
