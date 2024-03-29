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

#ifndef COLLIE_SIMD_ARCH_GENERIC_FWD_H_
#define COLLIE_SIMD_ARCH_GENERIC_FWD_H_

#include <collie/simd/types/batch_constant.h>

#include <type_traits>

namespace collie::simd {
    namespace kernel {
        // forward declaration
        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> abs(batch<T, A> const &self, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A>
        bitwise_lshift(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A>
        bitwise_rshift(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T>
        inline batch_bool<T, A> gt(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> mul(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> sadd(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> ssub(batch<T, A> const &self, batch<T, A> const &other, requires_arch<generic>) noexcept;

        template<class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline T hadd(batch<T, A> const &self, requires_arch<generic>) noexcept;

    }
}

#endif  // COLLIE_SIMD_ARCH_GENERIC_FWD_H_
