// Copyright 2024 The Elastic AI Search Authors.
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
#pragma once

#include <cstddef>
#include <type_traits>

namespace collie {

    template<typename T>
    constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, bool>
    is_range_invalid(T beg, T end, T step) {
        return ((step == 0 && beg != end) ||
                (beg < end && step <= 0) ||  // positive range
                (beg > end && step >= 0));   // negative range
    }

    template<typename T>
    constexpr std::enable_if_t<std::is_integral<std::decay_t<T>>::value, size_t>
    distance(T beg, T end, T step) {
        return (end - beg + step + (step > 0 ? -1 : 1)) / step;
    }

}  // end of namespace collie
