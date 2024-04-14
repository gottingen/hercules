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
// File: match.h
// -----------------------------------------------------------------------------
//
// This file contains simple utilities for performing string matching checks.
// All of these function parameters are specified as `std::string_view`,
// meaning that these functions can accept `std::string`, `std::string_view` or
// NUL-terminated C-style strings.
//
// Examples:
//   std::string s = "foo";
//   std::string_view sv = "f";
//   assert(turbo::str_contains(s, sv));
//
// Note: The order of parameters in these functions is designed to mimic the
// order an equivalent member function would exhibit;
// e.g. `s.Contains(x)` ==> `turbo::str_contains(s, x).
#ifndef TURBO_STRINGS_MATCH_H_
#define TURBO_STRINGS_MATCH_H_

#include <cstring>
#include <algorithm>
#include "turbo/strings/string_view.h"
#include "turbo/strings/ascii.h"

namespace turbo {

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::str_contains(input, "b"));
     *        @endcode
     * @param haystack The string to search in.
     * @param needle The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]] constexpr bool str_contains(std::string_view haystack,
                            std::string_view needle) noexcept {

        return haystack.find(needle, 0) != haystack.npos;
    }

    [[nodiscard]] constexpr bool str_contains(std::string_view haystack, char needle) noexcept {
        return haystack.find(needle) != haystack.npos;
    }

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle` ignore case.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::str_ignore_case_contains(input, "B"));
     *        @endcode
     * @param haystack The string to search in.
     * @param needle The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]] bool str_ignore_case_contains(std::string_view haystack,
                            std::string_view needle) noexcept;

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle` ignore case.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::str_ignore_case_contains(input, 'B'));
     *        @endcode
     * @note This function is constexpr if and only if the compiler supports.
     * @param haystack The string to search in.
     * @param needle The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]] constexpr bool str_ignore_case_contains(std::string_view haystack, char needle) noexcept {
        auto lc = turbo::ascii_to_lower(needle);
        auto uc = turbo::ascii_to_upper(needle);
        for(auto c : haystack) {
            if(c == lc || c == uc) {
                return true;
            }
        }
        return false;
    }


    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given string `text` begins with `prefix`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::starts_with(input, "a"));
     *        @endcode
     * @note This function is constexpr if and only if the compiler supports.
     * @param text The string to search in.
     * @param prefix The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    constexpr bool starts_with(std::string_view text,
                           std::string_view prefix) noexcept {
        return prefix.empty() ||
               (text.size() >= prefix.size() &&
                memcmp(text.data(), prefix.data(), prefix.size()) == 0);
    }

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given string `text` ends with `suffix`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::ends_with(input, "c"));
     *        @endcode
     * @param text The string to search in.
     * @param suffix The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    constexpr bool ends_with(std::string_view text,
                         std::string_view suffix) noexcept {
        return suffix.empty() ||
               (text.size() >= suffix.size() &&
                memcmp(text.data() + (text.size() - suffix.size()), suffix.data(),
                       suffix.size()) == 0);
    }

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether given ASCII strings `piece1` and `piece2` are equal, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::str_equals_ignore_case(input, "ABC"));
     *        @endcode
     * @note This function is constexpr if and only if the compiler supports.
     * @param piece1 The first string to compare.
     * @param piece2 The second string to compare.
     * @return true if the strings are equal, false otherwise.
     */
    [[nodiscard]]
    constexpr bool str_equals_ignore_case(const std::string_view &piece1,
                                          const std::string_view & piece2) noexcept {
        auto constexpr func =[](std::string_view piece1, std::string_view piece2) constexpr -> bool {
            for (std::size_t i = 0; i < piece1.size(); ++i) {
                if (ascii_to_lower(piece1[i]) != ascii_to_lower(piece2[i])) {
                    return false;
                }
            }
            return true;
        };
        return piece1.size() == piece2.size() && func(piece1, piece2);
    }

    bool str_equals_ignore_case(const wchar_t* lhs, const wchar_t* rhs);

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether given ASCII strings `first` and `last` are equal, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::str_equals_ignore_case(input.data(), input.end(), "ABC"));
     *        @endcode
     * @note This function is constexpr if and only if the compiler supports.
     * @param first The first string to compare.
     * @param last The last string to compare.
     * @param str The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]]
    constexpr bool str_equals_ignore_case(const char *first, const char *last, std::string_view str) noexcept {
        for(size_t i = 0; first != last && i != str.size(); ++first, ++i) {
            if(ascii_to_lower(*first) != ascii_to_lower(str[i])) {
                return false;
            }
        }
        return true;
    }

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given ASCII string `text` starts with `prefix`,
     *        ignoring case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::starts_with_ignore_case(input, "A"));
     *        @endcode
     * @param text The string to search in.
     * @param prefix The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    constexpr bool starts_with_ignore_case(std::string_view text,
                              std::string_view prefix) noexcept {
        return (text.size() >= prefix.size()) &&
               str_equals_ignore_case(text.substr(0, prefix.size()), prefix);
    }

    /**
     * @ingroup turbo_strings_match
     * @brief Returns whether a given ASCII string `text` ends with `suffix`, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(turbo::ends_with_ignore_case(input, "C"));
     *        @endcode
     * @param text The string to search in.
     * @param suffix The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    constexpr bool ends_with_ignore_case(std::string_view text,
                            std::string_view suffix) noexcept {
        return (text.size() >= suffix.size()) &&
               str_equals_ignore_case(text.substr(text.size() - suffix.size()), suffix);
    }

}  // namespace turbo

#endif  // TURBO_STRINGS_MATCH_H_
