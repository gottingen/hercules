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
#ifndef TURBO_STRINGS_STR_TRIM_H_
#define TURBO_STRINGS_STR_TRIM_H_

#include "turbo/strings/ascii.h"
#include "turbo/platform/port.h"

namespace turbo {

    /**
     * @ingroup turbo_strings_trim
     * @brief operator()() is used to check if a character is in the given string.
     */
    struct by_any_of {
        explicit by_any_of(std::string_view str) : trimmer(str) {

        }

        bool operator()(char c) {
            return trimmer.find(c) != std::string_view::npos;
        }

        bool operator()(unsigned char c) {
            return trimmer.find(c) != std::string_view::npos;
        }

    private:
        std::string_view trimmer;
    };

    /**
     * @ingroup turbo_strings_trim
     * @brief operator()() is used to check if a character is a whitespace.
     */
    struct by_white_space {
        by_white_space() = default;

        bool operator()(unsigned char c) {
            return turbo::ascii_is_space(c);
        }
    };

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_left() removes whitespace from the beginning of the given string.
     *         pred defaults to by_white_space.
     *         Example:
     *         @code
     *         std::string_view input("\t abc");
     *         EXPECT_EQ(turbo::trim_left(input), "abc");
     *         EXPECT_EQ(turbo::trim_left(input, turbo::by_any_of("\t a")), "bc");
     *         @endcode
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *        Defaults to by_white_space.
     * @return A std::string_view with whitespace stripped from the beginning of the
     *         given std::string_view.
     */
    template<typename Pred = by_white_space>
    [[nodiscard]] inline std::string_view trim_left(std::string_view str, Pred pred = Pred()) {
        auto it = std::find_if_not(str.begin(), str.end(), pred);
        return str.substr(static_cast<size_t>(it - str.begin()));
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_left() removes specified characters that match the predicate from the beginning of the given string.
     *        pred defaults to by_white_space.
     *        Example:
     *        @code
     *        std::string input("\t abc");
     *        turbo::trim_left(&input);
     *        EXPECT_EQ(input, "abc");
     *        std::string input2("\t abc");
     *        turbo::trim_left(&input2, turbo::by_any_of("\t a"));
     *        EXPECT_EQ(input2, "bc");
     *        @endcode
     * @attention This function will modify the given string.
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *       Defaults to by_white_space.
     */
    template<typename String, typename Pred = by_white_space>
    inline typename std::enable_if<turbo::is_string_type<String>::value>::type
    trim_left(String *str, Pred pred = Pred()) {
        auto it = std::find_if_not(str->begin(), str->end(), pred);
        str->erase(str->begin(), it);
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_right() removes whitespace from the end of the given string.
     *        pred defaults to by_white_space.
     *        Example:
     *        @code
     *        std::string_view input("abc \t");
     *        EXPECT_EQ(turbo::trim_right(input), "abc");
     *        EXPECT_EQ(turbo::trim_right(input, turbo::by_any_of("\t a")), "abc");
     *        @endcode
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *        Defaults to by_white_space.
     * @return A std::string_view with whitespace stripped from the end of the
     *         given std::string_view.
     */
    template<typename Pred = by_white_space>
    [[nodiscard]] inline std::string_view trim_right(std::string_view str, Pred pred = Pred()) {
        auto it = std::find_if_not(str.rbegin(), str.rend(), pred);
        return str.substr(0, static_cast<size_t>(str.rend() - it));
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_right() removes specified characters that match the predicate from the end of the given string.
     *        pred defaults to by_white_space.
     *        Example:
     *        @code
     *        std::string input("abc \t");
     *        turbo::trim_right(&input);
     *        EXPECT_EQ(input, "abc");
     *        std::string input2("abc \t");
     *        turbo::trim_right(&input2, turbo::by_any_of("\t a"));
     *        EXPECT_EQ(input2, "abc");
     *        @endcode
     * @attention This function will modify the given string.
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *        Defaults to by_white_space.
     */
    template<typename String, typename Pred = by_white_space>
    inline typename std::enable_if<turbo::is_string_type<String>::value>::type
    trim_right(String *str, Pred pred = Pred()) {
        auto it = std::find_if_not(str->rbegin(), str->rend(), pred);
        str->erase(static_cast<size_t>(str->rend() - it));
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_all() removes specified characters that match the predicate from both ends of the given string.
     *        pred defaults to by_white_space.
     *        Example:
     *        @code
     *        std::string_view input(" \t abc \t");
     *        EXPECT_EQ(turbo::trim_all(input), "abc");
     *        EXPECT_EQ(turbo::trim_all(input, turbo::by_any_of("\t a")), "bc");
     *        @endcode
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *       Defaults to by_white_space.
     * @return A std::string_view with whitespace stripped from both ends of the
     *        given std::string_view.
     */
    template<typename Pred = by_white_space>
    [[nodiscard]] inline std::string_view trim_all(std::string_view str, Pred pred = Pred()) {
        return trim_right(trim_left(str, pred), pred);
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_all() removes specified characters that match the predicate from both ends of the given string.
     *        pred defaults to by_white_space.
     *        Example:
     *        @code
     *        std::string input(" \t abc \t");
     *        turbo::trim_all(&input);
     *        EXPECT_EQ(input, "abc");
     *        std::string input2(" \t abc \t");
     *        turbo::trim_all(&input2, turbo::by_any_of("\t a"));
     *        EXPECT_EQ(input2, "bc");
     *        @endcode
     * @attention This function will modify the given string.
     * @param str The string to trim.
     * @param pred The predicate to use to determine if a character should be trimmed.
     *        Defaults to by_white_space.
     */
    template<typename String, typename Pred = by_white_space>
    inline typename std::enable_if<turbo::is_string_type<String>::value>::type
    trim_all(String *str, Pred pred = Pred()) {
        trim_right(str, pred);
        trim_left(str, pred);
    }

    /**
     * @ingroup turbo_strings_trim
     * @brief trim_complete() removes leading, trailing, and consecutive internal whitespace.
     *        Example:
     *        @code
     *        std::string input(" \t ab c \t");
     *        turbo::trim_complete(&input);
     *        EXPECT_EQ(input, "abc");
     *        @endcode
     * @attention This function will modify the given string.
     * @param str The string to trim.
     */
    template<typename String>
    typename std::enable_if<turbo::is_string_type<String>::value>::type
    trim_complete(String *);

}
#endif  // TURBO_STRINGS_STR_TRIM_H_
