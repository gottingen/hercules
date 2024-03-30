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

#include <type_traits>

namespace collie::ts::detail {
    // std::is_assignable but without user-defined conversions
    template<typename T, typename Arg>
    struct is_direct_assignable {
        template<typename U>
        struct consume_udc {
            operator U() const;
        };

        template<typename U>
        static std::true_type check(decltype(std::declval<T &>() = std::declval<consume_udc<U>>(),
                0) *);

        template<typename U>
        static std::false_type check(...);

        static constexpr bool value = decltype(check<Arg>(0))::value;
    };
} // namespace collie::ts::detail
