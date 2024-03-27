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

#ifndef COLLIE_STRINGS_MATCH_H_
#define COLLIE_STRINGS_MATCH_H_

#include <cstring>
#include <algorithm>
#include <string>
#include <string_view>
#include <collie/strings/ascii.h>
#include <collie/strings/case_conv.h>

namespace collie {

    /**
     * @ingroup collie_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::str_contains(input, "b"));
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
     * @ingroup collie_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle` ignore case.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::str_ignore_case_contains(input, "B"));
     *        @endcode
     * @param haystack The string to search in.
     * @param needle The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]] inline bool str_ignore_case_contains(std::string_view haystack,
                            std::string_view needle) noexcept {
        auto ih = str_to_lower(haystack);
        auto in = str_to_lower(needle);
        return str_contains(ih, in);
    }

    /**
     * @ingroup collie_strings_match
     * @brief Returns whether a given string `haystack` contains the substring `needle` ignore case.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::str_ignore_case_contains(input, 'B'));
     *        @endcode
     * @note This function is constexpr if and only if the compiler supports.
     * @param haystack The string to search in.
     * @param needle The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    [[nodiscard]] constexpr bool str_ignore_case_contains(std::string_view haystack, char needle) noexcept {
        auto lc = collie::ascii_to_lower(needle);
        auto uc = collie::ascii_to_upper(needle);
        for(auto c : haystack) {
            if(c == lc || c == uc) {
                return true;
            }
        }
        return false;
    }


    /**
     * @ingroup collie_strings_match
     * @brief Returns whether a given string `text` begins with `prefix`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::starts_with(input, "a"));
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
     * @ingroup collie_strings_match
     * @brief Returns whether a given string `text` ends with `suffix`.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::ends_with(input, "c"));
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
     * @ingroup collie_strings_match
     * @brief Returns whether given ASCII strings `piece1` and `piece2` are equal, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::str_equals_ignore_case(input, "ABC"));
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

    bool str_equals_ignore_case(const wchar_t* lhs, const wchar_t* rhs) {
        if (lhs == nullptr) return rhs == nullptr;

        if (rhs == nullptr) return false;

#ifdef _WIN32
        return _wcsicmp(lhs, rhs) == 0;
#elif defined(__linux__) && !defined(__android__)
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

    /**
     * @ingroup collie_strings_match
     * @brief Returns whether given ASCII strings `first` and `last` are equal, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::str_equals_ignore_case(input.data(), input.end(), "ABC"));
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
     * @ingroup collie_strings_match
     * @brief Returns whether a given ASCII string `text` starts with `prefix`,
     *        ignoring case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::starts_with_ignore_case(input, "A"));
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
     * @ingroup collie_strings_match
     * @brief Returns whether a given ASCII string `text` ends with `suffix`, ignoring
     *        case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::ends_with_ignore_case(input, "C"));
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

    /**
     * @ingroup collie_strings_match
     * @brief Returns whether a given ASCII string `text` starts with `prefix`,
     *        ignoring case in the comparison.
     *        Example:
     *        @code
     *        std::string_view input("abc");
     *        EXPECT_TRUE(collie::starts_with_ignore_case(input, 'A'));
     *        @endcode
     * @param text The string to search in.
     * @param prefix The substring to search for.
     * @return true if the substring is found, false otherwise.
     */
    struct CharCompareBase {
        CharCompareBase() = default;

        constexpr operator bool() const { return ptr != nullptr; }

        constexpr  bool operator==(char c) const {
            return ptr && *ptr == c;
        }

        constexpr  bool operator!=(char c) const {
            return ptr && *ptr != c;
        }

        constexpr  bool operator<(char c) const {
            return ptr && *ptr < c;
        }

        constexpr  bool operator>(char c) const {
            return ptr && *ptr > c;
        }

        constexpr  bool operator<=(char c) const {
            return ptr && *ptr <= c;
        }

        constexpr  bool operator>=(char c) const {
            return ptr && *ptr >= c;
        }

        constexpr const char* value() const {
            return ptr;
        }

        const char *ptr{nullptr};
    };


    struct BackChar : public CharCompareBase {
        BackChar(std::string_view str) {
            if (!str.empty()) {
                ptr = &str.back();
            }
        }
    };

    struct FrontChar : public CharCompareBase {
        FrontChar(std::string_view str) {
            if (!str.empty()) {
                ptr = &str.front();
            }
        }
    };

}  // namespace collie

#endif  // COLLIE_STRINGS_MATCH_H_
