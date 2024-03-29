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
    template<bool... Bs>
    struct bool_sequence {
    };

    template<bool... Bs>
    using all_of = std::is_same<bool_sequence<Bs...>, bool_sequence<(true || Bs)...>>;

    template<bool... Bs>
    using none_of = std::is_same<bool_sequence<Bs...>, bool_sequence<(false && Bs)...>>;

    template<bool... Bs>
    using any_of = std::integral_constant<bool, !none_of<Bs...>::value>;
} // namespace collie::ts::detail
