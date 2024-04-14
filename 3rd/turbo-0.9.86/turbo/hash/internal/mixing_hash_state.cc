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

#include "turbo/hash/internal/hash.h"
#include "turbo/hash/bytes/bytes_hash.h"

namespace turbo::hash_internal {

    template<typename Tag>
    uint64_t MixingHashState<Tag>::CombineLargeContiguousImpl32(
            uint64_t state, const unsigned char *first, size_t len) {
        while (len >= PiecewiseChunkSize()) {
            state = hasher_engine<Tag>::hash32_with_seed(reinterpret_cast<const char *>(first), PiecewiseChunkSize(), state);
            len -= PiecewiseChunkSize();
            first += PiecewiseChunkSize();
        }
        // Handle the remainder.
        return len ? hasher_engine<Tag>::hash32_with_seed(reinterpret_cast<const char *>(first), len, state) : state;
    }

    template<typename Tag>
    uint64_t MixingHashState<Tag>::CombineLargeContiguousImpl64(
            uint64_t state, const unsigned char *first, size_t len) {
        while (len >= PiecewiseChunkSize()) {
            state = hasher_engine<Tag>::hash64_with_seed(reinterpret_cast<const char *>(first), PiecewiseChunkSize(), state);
            len -= PiecewiseChunkSize();
            first += PiecewiseChunkSize();
        }
        // Handle the remainder.
        return len ? hasher_engine<Tag>::hash64_with_seed(reinterpret_cast<const char *>(first), len, state) : state;
    }

    template class MixingHashState<bytes_hash_tag>;
    template class MixingHashState<m3_hash_tag>;
    template class MixingHashState<xx_hash_tag>;
}  // namespace turbo::hash_internal
