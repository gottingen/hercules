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

#ifndef COLLIE_SIMD_ARCH_SSE4_2_H_
#define COLLIE_SIMD_ARCH_SSE4_2_H_

#include <limits>

#include <collie/simd/types/sse4_2_register.h>

namespace collie::simd {

    namespace kernel {
        using namespace types;

        // lt
        template<class A>
        inline batch_bool<int64_t, A>
        lt(batch<int64_t, A> const &self, batch<int64_t, A> const &other, requires_arch<sse4_2>) noexcept {
            return _mm_cmpgt_epi64(other, self);
        }

        template<class A>
        inline batch_bool<uint64_t, A>
        lt(batch<uint64_t, A> const &self, batch<uint64_t, A> const &other, requires_arch<sse4_2>) noexcept {
            auto xself = _mm_xor_si128(self, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
            auto xother = _mm_xor_si128(other, _mm_set1_epi64x(std::numeric_limits<int64_t>::lowest()));
            return _mm_cmpgt_epi64(xother, xself);
        }

    }

}  // namespace collie::simd

#endif  // COLLIE_SIMD_ARCH_SSE4_2_H_
