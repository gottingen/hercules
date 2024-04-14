// Copyright 2020 The Turbo Authors.
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

#include "turbo/strings/match.h"
#include "turbo/strings/str_case_conv.h"
#include "turbo/strings/internal/memutil.h"

namespace turbo {

    bool str_ignore_case_contains(std::string_view haystack,
                           std::string_view needle) noexcept {
        auto ih = str_to_lower(haystack);
        auto in = str_to_lower(needle);
        return str_contains(ih, in);
    }

    bool str_equals_ignore_case(const wchar_t* lhs, const wchar_t* rhs) {
        if (lhs == nullptr) return rhs == nullptr;

        if (rhs == nullptr) return false;

#ifdef TURBO_PLATFORM_WINDOWS
        return _wcsicmp(lhs, rhs) == 0;
#elif defined(TURBO_PLATFORM_LINUX) && !defined(TURBO_PLATFORM_ANDROID)
        return wcscasecmp(lhs, rhs) == 0;
#else
        // Android, Mac OS X and Cygwin don't define wcscasecmp.
        // Other unknown OSes may not define it either.
        wint_t left, right;
        do {
            left = towlower(static_cast<wint_t>(*lhs++));
            right = towlower(static_cast<wint_t>(*rhs++));
        } while (left && left == right);
        return left == right;
#endif
    }

}  // namespace turbo
