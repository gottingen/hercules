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
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>


#include "turbo/platform/port.h"

namespace turbo {


    /// Include the items in this namespace to get free conversion of enums to/from streams.
    /// (This is available inside flags as well, so FLAGS will use this without a using statement).
    namespace enums {

        /// output streaming for enumerations
        template<typename T, typename = typename std::enable_if<std::is_enum<T>::value>::type>
        std::ostream &operator<<(std::ostream &in, const T &item) {
            // make sure this is out of the detail namespace otherwise it won't be found when needed
            return in << static_cast<typename std::underlying_type<T>::type>(item);
        }

    }  // namespace enums

        /// Export to flags namespace
    using enums::operator<<;

    namespace detail {
        /// a constant defining an expected max vector size defined to be a big number that could be multiplied by 4 and not
        /// produce overflow for some expected uses
        constexpr int expected_max_vector_size{1 << 29};

        // Based on http://stackoverflow.com/questions/236129/split-a-string-in-c
        /// Split a string by a delim
        std::vector<std::string> split(const std::string &s, char delim);

        /// Simple function to join a string
        template<typename T>
        std::string join(const T &v, std::string delim = ",") {
            std::ostringstream s;
            auto beg = std::begin(v);
            auto end = std::end(v);
            if (beg != end)
                s << *beg++;
            while (beg != end) {
                s << delim << *beg++;
            }
            return s.str();
        }

        /// Simple function to join a string from processed elements
        template<typename T,
                typename Callable,
                typename = typename std::enable_if<!std::is_constructible<std::string, Callable>::value>::type>
        std::string join(const T &v, Callable func, std::string delim = ",") {
            std::ostringstream s;
            auto beg = std::begin(v);
            auto end = std::end(v);
            auto loc = s.tellp();
            while (beg != end) {
                auto nloc = s.tellp();
                if (nloc > loc) {
                    s << delim;
                    loc = nloc;
                }
                s << func(*beg++);
            }
            return s.str();
        }

        /// Join a string in reverse order
        template<typename T>
        std::string rjoin(const T &v, std::string delim = ",") {
            std::ostringstream s;
            for (std::size_t start = 0; start < v.size(); start++) {
                if (start > 0)
                    s << delim;
                s << v[v.size() - start - 1];
            }
            return s.str();
        }

        // Based roughly on http://stackoverflow.com/questions/25829143/c-trim-whitespace-from-a-string

        /// trim_all whitespace from left of string
        std::string &ltrim(std::string &str);

        /// trim_all anything from left of string
        std::string &ltrim(std::string &str, const std::string &filter);

        /// trim_all whitespace from right of string
        std::string &rtrim(std::string &str);

        /// trim_all anything from right of string
        std::string &rtrim(std::string &str, const std::string &filter);

        /// trim_all whitespace from string
        inline std::string &trim(std::string &str) { return ltrim(rtrim(str)); }

        /// trim_all anything from string
        inline std::string &trim(std::string &str, const std::string filter) {
            return ltrim(rtrim(str, filter), filter);
        }

        /// Make a copy of the string and then trim it
        inline std::string trim_copy(const std::string &str) {
            std::string s = str;
            return trim(s);
        }

        /// remove quotes at the front and back of a string either '"' or '\''
        std::string &remove_quotes(std::string &str);

        /// Add a leader to the beginning of all new lines (nothing is added
        /// at the start of the first line). `"; "` would be for ini files
        ///
        /// Can't use Regex, or this would be a subs.
        std::string fix_newlines(const std::string &leader, std::string input);

        /// Make a copy of the string and then trim it, any filter string can be used (any char in string is filtered)
        inline std::string trim_copy(const std::string &str, const std::string &filter) {
            std::string s = str;
            return trim(s, filter);
        }

        /// Print a two part "help" string
        std::ostream &
        format_help(std::ostream &out, std::string name, const std::string &description, std::size_t wid);

        /// Print subcommand aliases
        std::ostream &format_aliases(std::ostream &out, const std::vector<std::string> &aliases, std::size_t wid);

        /// Verify the first character of an option
        /// - is a trigger character, ! has special meaning and new lines would just be annoying to deal with
        template<typename T>
        bool valid_first_char(T c) { return ((c != '-') && (c != '!') && (c != ' ') && c != '\n'); }

        /// Verify following characters of an option
        template<typename T>
        bool valid_later_char(T c) {
            // = and : are value separators, { has special meaning for option defaults,
            // and \n would just be annoying to deal with in many places allowing space here has too much potential for
            // inadvertent entry errors and bugs
            return ((c != '=') && (c != ':') && (c != '{') && (c != ' ') && c != '\n');
        }

        /// Verify an option/subcommand name
        bool valid_name_string(const std::string &str);

        /// Verify an app name
        inline bool valid_alias_name_string(const std::string &str) {
            static const std::string badChars(std::string("\n") + '\0');
            return (str.find_first_of(badChars) == std::string::npos);
        }

        /// check if a string is a container segment separator (empty or "%%")
        inline bool is_separator(const std::string &str) {
            static const std::string sep("%%");
            return (str.empty() || str == sep);
        }

        /// Verify that str consists of letters only
        inline bool isalpha(const std::string &str) {
            return std::all_of(str.begin(), str.end(), [](char c) { return std::isalpha(c, std::locale()); });
        }

        /// Return a lower case version of a string
        inline std::string to_lower(std::string str) {
            std::transform(std::begin(str), std::end(str), std::begin(str), [](const std::string::value_type &x) {
                return std::tolower(x, std::locale());
            });
            return str;
        }

        /// remove underscores from a string
        inline std::string remove_underscore(std::string str) {
            str.erase(std::remove(std::begin(str), std::end(str), '_'), std::end(str));
            return str;
        }

        /// Find and replace a substring with another substring
        std::string find_and_replace(std::string str, std::string from, std::string to);

        /// check if the flag definitions has possible false flags
        inline bool has_default_flag_values(const std::string &flags) {
            return (flags.find_first_of("{!") != std::string::npos);
        }

        void remove_default_flag_values(std::string &flags);

        /// Check if a string is a member of a list of strings and optionally ignore case or ignore underscores
        std::ptrdiff_t find_member(std::string name,
                                   const std::vector<std::string> names,
                                   bool ignore_case = false,
                                   bool ignore_underscore = false);

        /// Find a trigger string and call a modify callable function that takes the current string and starting position of the
        /// trigger and returns the position in the string to search for the next trigger string
        template<typename Callable>
        inline std::string find_and_modify(std::string str, std::string trigger, Callable modify) {
            std::size_t start_pos = 0;
            while ((start_pos = str.find(trigger, start_pos)) != std::string::npos) {
                start_pos = modify(str, start_pos);
            }
            return str;
        }

        /// Split a string '"one two" "three"' into 'one two', 'three'
        /// Quote characters can be ` ' or "
        std::vector<std::string> split_up(std::string str, char delimiter = '\0');

        /// This function detects an equal or colon followed by an escaped quote after an argument
        /// then modifies the string to replace the equality with a space.  This is needed
        /// to allow the split up function to work properly and is intended to be used with the find_and_modify function
        /// the return value is the offset+1 which is required by the find_and_modify function.
        std::size_t escape_detect(std::string &str, std::size_t offset);

        /// Add quotes if the string contains spaces
        std::string &add_quotes_if_needed(std::string &str);

    }  // namespace detail


}  // namespace turbo
