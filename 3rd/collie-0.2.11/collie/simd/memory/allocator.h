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

#ifndef COLLIE_SIMD_MEMORY_ALLOCATOR_H_
#define COLLIE_SIMD_MEMORY_ALLOCATOR_H_

#include <collie/memory/aligned_allocator.h>
#include <collie/simd/types/simd_utils.h>
#include <collie/simd/config/simd_arch.h>

namespace collie::simd {
    template <class T, class A = default_arch>
    using default_allocator = typename std::conditional<A::requires_alignment(),
                                                        collie::aligned_allocator<T, A::alignment()>,
                                                        std::allocator<T>>::type;

    /**
  * @struct aligned_mode
  * @brief tag for load and store of aligned memory.
  */
    struct aligned_mode {
    };

    /**
     * @struct unaligned_mode
     * @brief tag for load and store of unaligned memory.
     */
    struct unaligned_mode {
    };

    /***********************
     * Allocator alignment *
     ***********************/

    template<class A>
    struct allocator_alignment {
        using type = unaligned_mode;
    };

    template<class T, size_t N>
    struct allocator_alignment<collie::aligned_allocator<T, N>> {
        using type = aligned_mode;
    };

    template<class A>
    using allocator_alignment_t = typename allocator_alignment<A>::type;

    /***********************
     * container alignment *
     ***********************/

    template<class C, class = void>
    struct container_alignment {
        using type = unaligned_mode;
    };

    template<class C>
    struct container_alignment<C, detail::void_t<typename C::allocator_type>> {
        using type = allocator_alignment_t<typename C::allocator_type>;
    };

    template<class C>
    using container_alignment_t = typename container_alignment<C>::type;

    /*********************
     * alignment checker *
     *********************/

    /**
     * Checks whether pointer \c ptr is aligned according the alignment
     * requirements of \c Arch.
     * @return true if the alignment requirements are met
     */
    template<class Arch = default_arch>
    inline bool is_aligned(void const *ptr) {
        return (reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(Arch::alignment())) == 0;
    }
}  // namespace collie::simd

#endif  // COLLIE_SIMD_MEMORY_ALLOCATOR_H_
