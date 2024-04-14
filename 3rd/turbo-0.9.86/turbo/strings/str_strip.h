//
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
//
// -----------------------------------------------------------------------------
// File: strip.h
// -----------------------------------------------------------------------------
//
// This file contains various functions for stripping substrings from a string.
#ifndef TURBO_STRINGS_STRIP_H_
#define TURBO_STRINGS_STRIP_H_

#include <cstddef>
#include <string>

#include "turbo/platform/port.h"
#include "turbo/strings/ascii.h"
#include "turbo/strings/match.h"
#include "turbo/strings/string_view.h"

namespace turbo {
    /**
     * @ingroup turbo_strings_trim
     * @brief Strips the `expected` prefix, if found, from the start of `str`.
     *        If the operation succeeded, `true` is returned.  If not, `false`
     *        is returned and `str` is not modified.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::consume_prefix(&input, "a"));
     *        EXPECT_EQ(input, "bc");
     *        @endcode
     * @param str The string to strip from. After the function returns, this
     *            will be the remaining string after the prefix is stripped.
     * @param expected The prefix to strip.
     * @return true if the prefix is stripped successfully, false otherwise.
     */
    inline bool consume_prefix(std::string_view *str, std::string_view expected) {
        if (!turbo::starts_with(*str, expected)) return false;
        str->remove_prefix(expected.size());
        return true;
    }

    // consume_suffix()
    //
    // Strips the `expected` suffix, if found, from the end of `str`.
    // If the operation succeeded, `true` is returned.  If not, `false`
    // is returned and `str` is not modified.
    //
    // Example:
    //
    //   std::string_view input("abcdef");
    //   EXPECT_TRUE(turbo::consume_suffix(&input, "def"));
    //   EXPECT_EQ(input, "abc");
    /**
     * @ingroup turbo_strings_trim
     * @brief Strips the `expected` suffix, if found, from the end of `str`.
     *        If the operation succeeded, `true` is returned.  If not, `false`
     *        is returned and `str` is not modified.
     *        Example:
     *        @code
     *        std::string_view input("abcdef");
     *        EXPECT_TRUE(turbo::consume_suffix(&input, "def"));
     *        EXPECT_EQ(input, "abc");
     *        @endcode
     * @param str The string to strip from. After the function returns, this
     *        will be the remaining string after the suffix is stripped.
     * @param expected The suffix to strip.
     * @return true if the suffix is stripped successfully, false otherwise.
     */
    inline bool consume_suffix(std::string_view *str, std::string_view expected) {
        if (!turbo::ends_with(*str, expected)) return false;
        str->remove_suffix(expected.size());
        return true;
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief Returns a view into the input string `str` with the given `prefix` removed,
     *        but leaving the original string intact. If the prefix does not match at the
     *        start of the string, returns the original string instead.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_EQ(turbo::strip_prefix(input, "a"), "bc");
     *        EXPECT_EQ(turbo::strip_prefix(input, "b"), "abc");
     *        @endcode
     * @param str The string to strip from.
     * @param prefix The prefix to strip.
     * @return A view into the input string `str` with the given `prefix` removed.
     */
    [[nodiscard]] inline std::string_view strip_prefix(
            std::string_view str, std::string_view prefix) {
        if (turbo::starts_with(str, prefix)) str.remove_prefix(prefix.size());
        return str;
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief Returns a view into the input string `str` with the given `suffix` removed,
     *        but leaving the original string intact. If the suffix does not match at the
     *        end of the string, returns the original string instead.
     *        Example:
     *        @code
     *        std::string_view input("abcdef");
     *        EXPECT_EQ(turbo::strip_suffix(input, "def"), "abc");
     *        EXPECT_EQ(turbo::strip_suffix(input, "de"), "abcdef");
     *        @endcode
     * @param str The string to strip from.
     * @param suffix The suffix to strip.
     * @return A view into the input string `str` with the given `suffix` removed.
     */
    [[nodiscard]] inline std::string_view strip_suffix(
            std::string_view str, std::string_view suffix) {
        if (turbo::ends_with(str, suffix)) str.remove_suffix(suffix.size());
        return str;
    }
}  // namespace turbo

#endif  // TURBO_STRINGS_STRIP_H_
