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

#ifndef COLLIE_STRINGS_CASE_CONV_H_
#define COLLIE_STRINGS_CASE_CONV_H_

#include <string>
#include <string_view>
#include <collie/strings/ascii.h>
#include <collie/strings/inlined_string.h>

namespace collie {

    template<typename String>
    struct is_string_type : std::false_type {};

    template<>
    struct is_string_type<std::string> : std::true_type {};

    template<unsigned N>
    struct is_string_type<collie::InlinedString<N>> : std::true_type {};

    /**
     * @ingroup collie_string_convert
     * @brief Converts the characters in `s` to lowercase, changing the contents of `s`.
     *        This function is a no-op if `String` is not a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        collie::str_to_lower(&s);
     *        assert(s == "hello world!");
     *        @endcode
     * @param s The string to convert to lowercase.
     */
    template<typename String>
    inline typename std::enable_if_t<is_string_type<String>::value> str_to_lower(String *s) {
        if (s->empty()) {
            return;
        }
        for (auto &c : *s) {
            c = collie::ascii_to_lower(c);
        }
    }

    /**
     * @ingroup collie_string_convert
     * @brief Creates a lowercase string from a given std::string_view.
     *        String is a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        std::string s1 = collie::str_to_lower(s);
     *        assert(s1 == "hello world!");
     *        @endcode
     * @param s The string to convert to lowercase.
     */
    inline std::string str_to_lower(std::string_view s) {
        std::string result;
        result.reserve(s.size());
        for (auto c : s) {
            result.append(1, collie::ascii_to_lower(c));
        }
        return result;
    }

    /**
     * @ingroup collie_string_convert
     * @brief str_to_upper Converts the characters in `s` to uppercase, changing the contents of `s`.
     *        This function is a no-op if `String` is not a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        collie::str_to_upper(&s);
     *        assert(s == "HELLO WORLD!");
     *        @endcode
     * @param s The string to convert to uppercase.
     */
    template<typename String>
    typename std::enable_if_t<is_string_type<String>::value> str_to_upper(String *s) {
        if (s->empty()) {
            return;
        }
        for (auto &c : *s) {
            c = collie::ascii_to_upper(c);
        }
    }

    /**
     * @ingroup collie_string_convert
     * @brief Creates an uppercase string from a given std::string_view.
     *        String is a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        std::string s1 = collie::str_to_upper(s);
     *        assert(s1 == "HELLO WORLD!");
     *        @endcode
     * @param s The string to convert to uppercase.
     */
    inline std::string str_to_upper(std::string_view s) {
          std::string result;
            result.reserve(s.size());
            for (auto c : s) {
                result.append(1, collie::ascii_to_upper(c));
            }
            return result;
    }

}  // namespace collie

#endif  // COLLIE_STRINGS_CASE_CONV_H_
