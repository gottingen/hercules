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

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include <clocale>
#include <locale>

#include <cstdlib>
#include <wchar.h>

namespace turbo {

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)

    inline int get_wcswidth(const std::string &string, const std::string &locale,
                            size_t max_column_width) {
        if (string.size() == 0)
            return 0;

        // The behavior of wcswidth() depends on the LC_CTYPE category of the current
        // locale. Set the current locale based on cell properties before computing
        // width
        auto old_locale = std::locale::global(std::locale(locale));

        // Convert from narrow std::string to wide string
        wchar_t *wide_string = new wchar_t[string.size()];
        std::mbstowcs(wide_string, string.c_str(), string.size());

        // Compute display width of wide string
        int result = wcswidth(wide_string, max_column_width);
        delete[] wide_string;

        // Restore old locale
        std::locale::global(old_locale);

        return result;
    }

#endif

    inline size_t get_sequence_length(const std::string &text, const std::string &locale,
                                      bool is_multi_byte_character_support_enabled) {
        if (!is_multi_byte_character_support_enabled)
            return text.length();

#if defined(_WIN32) || defined(_WIN64)
        (void)locale; // unused parameter
        return (text.length() - std::count_if(text.begin(), text.end(),
                                              [](char c) -> bool { return (c & 0xC0) == 0x80; }));
#elif defined(__unix__) || defined(__unix) || defined(__APPLE__)
        auto result = get_wcswidth(text, locale, text.size());
        if (result >= 0)
            return result;
        else
            return (text.length() - std::count_if(text.begin(), text.end(),
                                                  [](char c) -> bool { return (c & 0xC0) == 0x80; }));
#endif
    }

} // namespace turbo
