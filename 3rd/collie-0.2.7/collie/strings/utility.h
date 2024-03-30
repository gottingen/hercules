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

#ifndef COLLIE_STRINGS_UTILITY_H_
#define COLLIE_STRINGS_UTILITY_H_

#include <string>
#include <string_view>

namespace collie {

    std::string_view safe_substring(std::string_view str, size_t pos, size_t len) {
        if (pos >= str.size()) {
            return std::string_view();
        }
        return str.substr(pos, len);
    }

    std::string_view safe_slice(std::string_view str, size_t start, size_t end) {
        if (start >= str.size() || start >= end) {
            return std::string_view();
        }
        return str.substr(start, end - start);
    }

}  // namespace collie
#endif  // COLLIE_STRINGS_UTILITY_H_
