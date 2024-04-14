//
// Copyright 2023 The Turbo Authors.
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
#include "turbo/strings/str_trim.h"
#include "turbo/strings/inlined_string.h"
#include "turbo/strings/ascii.h"

namespace turbo {

    template<typename String>
    typename std::enable_if<turbo::is_string_type<String>::value>::type
    trim_complete(String *str) {
        auto stripped = trim_all(*str);

        if (stripped.empty()) {
            str->clear();
            return;
        }

        auto input_it = stripped.begin();
        auto input_end = stripped.end();
        auto output_it = &(*str)[0];
        bool is_ws = false;

        for (; input_it < input_end; ++input_it) {
            if (is_ws) {
                // Consecutive whitespace?  Keep only the last.
                is_ws = turbo::ascii_is_space(static_cast<unsigned char>(*input_it));
                if (is_ws) --output_it;
            } else {
                is_ws = turbo::ascii_is_space(static_cast<unsigned char>(*input_it));
            }

            *output_it = *input_it;
            ++output_it;
        }

        str->erase(static_cast<size_t>(output_it - &(*str)[0]));
    }

    template void trim_complete(std::string *str);

    template void trim_complete(turbo::inlined_string *str);

}  // namespace turbo
