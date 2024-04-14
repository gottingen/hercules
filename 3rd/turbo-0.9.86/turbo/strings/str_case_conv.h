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
#ifndef TURBO_STRINGS_STR_CASE_CONV_H_
#define TURBO_STRINGS_STR_CASE_CONV_H_
#include "turbo/platform/port.h"
#include "turbo/meta/type_traits.h"

namespace turbo {

    /**
     * @ingroup turbo_string_convert
     * @brief Converts the characters in `s` to lowercase, changing the contents of `s`.
     *        This function is a no-op if `String` is not a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        turbo::str_to_lower(&s);
     *        assert(s == "hello world!");
     *        @endcode
     * @param s The string to convert to lowercase.
     */
    template<typename String>
    TURBO_MUST_USE_RESULT typename std::enable_if<turbo::is_string_type<String>::value>::type
    str_to_lower(String *s);


    /**
     * @ingroup turbo_string_convert
     * @brief Creates a lowercase string from a given std::string_view.
     *        String is a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        std::string s1 = turbo::str_to_lower(s);
     *        assert(s1 == "hello world!");
     *        @endcode
     * @param s The string to convert to lowercase.
     */
    template<typename String = std::string>
    TURBO_MUST_USE_RESULT inline typename std::enable_if<turbo::is_string_type<String>::value, String>::type
    str_to_lower(std::string_view s) {
        String result(s);
        turbo::str_to_lower(&result);
        return result;
    }

    /**
     * @ingroup turbo_string_convert
     * @brief str_to_upper Converts the characters in `s` to uppercase, changing the contents of `s`.
     *        This function is a no-op if `String` is not a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        turbo::str_to_upper(&s);
     *        assert(s == "HELLO WORLD!");
     *        @endcode
     * @param s The string to convert to uppercase.
     */
    template<typename String>
    TURBO_MUST_USE_RESULT typename std::enable_if<turbo::is_string_type<String>::value>::type
    str_to_upper(String *s);

    /**
     * @ingroup turbo_string_convert
     * @brief Creates an uppercase string from a given std::string_view.
     *        String is a string type.
     *        Example:
     *        @code
     *        std::string s = "Hello World!";
     *        std::string s1 = turbo::str_to_upper(s);
     *        assert(s1 == "HELLO WORLD!");
     *        @endcode
     * @param s The string to convert to uppercase.
     */
    template<typename String = std::string>
    TURBO_MUST_USE_RESULT inline typename std::enable_if<turbo::is_string_type<String>::value, String>::type
    str_to_upper(std::string_view s) {
        String result(s);
        turbo::str_to_upper(&result);
        return result;
    }

}  // namespace turbo

#endif  // TURBO_STRINGS_STR_CASE_CONV_H_
