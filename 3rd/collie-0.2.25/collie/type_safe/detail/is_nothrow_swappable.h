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
//

#pragma once

#include <utility>

namespace collie::ts::detail {
    template<typename T>
    struct is_nothrow_swappable {
        template<typename U>
        static auto adl_swap(int, U &a, U &b) noexcept(noexcept(swap(a, b)))
        -> decltype(swap(a, b));

        template<typename U>
        static auto adl_swap(short, U &a, U &b) noexcept(noexcept(std::swap(a, b)))
        -> decltype(std::swap(a, b));

        static void adl_swap(...) noexcept(false);

        static constexpr bool value = noexcept(adl_swap(0, std::declval<T &>(), std::declval<T &>()));
    };
} // namespace collie::ts::detail
