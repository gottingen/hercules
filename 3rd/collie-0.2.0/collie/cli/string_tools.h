// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <collie/cli/macros.h>

namespace collie {


    /// Include the items in this namespace to get free conversion of enums to/from streams.
    namespace enums {

        /// output streaming for enumerations
        template<typename T, typename = typename std::enable_if<std::is_enum<T>::value>::type>
        std::ostream &operator<<(std::ostream &in, const T &item) {
            // make sure this is out of the detail namespace otherwise it won't be found when needed
            return in << static_cast<typename std::underlying_type<T>::type>(item);
        }

    }  // namespace enums

    /// Export to CLI namespace
    using enums::operator<<;

    namespace detail {
        /// a constant defining an expected max vector size defined to be a big number that could be multiplied by 4 and not
        /// produce overflow for some expected uses
        constexpr int expected_max_vector_size{1 << 29};

        // Based on http://stackoverflow.com/questions/236129/split-a-string-in-c
        /// Split a string by a delim
        inline std::vector<std::string> split(const std::string &s, char delim);

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

        /// Trim whitespace from left of string
        inline std::string &ltrim(std::string &str);

        /// Trim anything from left of string
        inline std::string &ltrim(std::string &str, const std::string &filter);

        /// Trim whitespace from right of string
        inline std::string &rtrim(std::string &str);

        /// Trim anything from right of string
        inline std::string &rtrim(std::string &str, const std::string &filter);

        /// Trim whitespace from string
        inline std::string &trim(std::string &str) { return ltrim(rtrim(str)); }

        /// Trim anything from string
        inline std::string &trim(std::string &str, const std::string filter) {
            return ltrim(rtrim(str, filter), filter);
        }

        /// Make a copy of the string and then trim it
        inline std::string trim_copy(const std::string &str) {
            std::string s = str;
            return trim(s);
        }

        /// remove quotes at the front and back of a string either '"' or '\''
        inline std::string &remove_quotes(std::string &str);

        /// remove quotes from all elements of a string vector and process escaped components
        inline void remove_quotes(std::vector<std::string> &args);

        /// Add a leader to the beginning of all new lines (nothing is added
        /// at the start of the first line). `"; "` would be for ini files
        ///
        /// Can't use Regex, or this would be a subs.
        inline std::string fix_newlines(const std::string &leader, std::string input);

        /// Make a copy of the string and then trim it, any filter string can be used (any char in string is filtered)
        inline std::string trim_copy(const std::string &str, const std::string &filter) {
            std::string s = str;
            return trim(s, filter);
        }

        /// Print a two part "help" string
        inline std::ostream &
        format_help(std::ostream &out, std::string name, const std::string &description, std::size_t wid);

        /// Print subcommand aliases
        inline std::ostream &
        format_aliases(std::ostream &out, const std::vector<std::string> &aliases, std::size_t wid);

        /// Verify the first character of an option
        /// - is a trigger character, ! has special meaning and new lines would just be annoying to deal with
        template<typename T>
        bool valid_first_char(T c) {
            return ((c != '-') && (static_cast<unsigned char>(c) > 33));  // space and '!' not allowed
        }

        /// Verify following characters of an option
        template<typename T>
        bool valid_later_char(T c) {
            // = and : are value separators, { has special meaning for option defaults,
            // and control codes other than tab would just be annoying to deal with in many places allowing space here has too
            // much potential for inadvertent entry errors and bugs
            return ((c != '=') && (c != ':') && (c != '{') && ((static_cast<unsigned char>(c) > 32) || c == '\t'));
        }

        /// Verify an option/subcommand name
        inline bool valid_name_string(const std::string &str);

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
        inline std::string find_and_replace(std::string str, std::string from, std::string to);

        /// check if the flag definitions has possible false flags
        inline bool has_default_flag_values(const std::string &flags) {
            return (flags.find_first_of("{!") != std::string::npos);
        }

        inline void remove_default_flag_values(std::string &flags);

        /// Check if a string is a member of a list of strings and optionally ignore case or ignore underscores
        inline std::ptrdiff_t find_member(std::string name,
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

        /// close a sequence of characters indicated by a closure character.  Brackets allows sub sequences
        /// recognized bracket sequences include <'`[(<{  other closure characters are assumed to be literal strings
        inline std::size_t close_sequence(const std::string &str, std::size_t start, char closure_char);

        /// Split a string '"one two" "three"' into 'one two', 'three'
        /// Quote characters can be ` ' or " or bracket characters [{(< with matching to the matching bracket
        inline std::vector<std::string> split_up(std::string str, char delimiter = '\0');

        /// get the value of an environmental variable or empty string if empty
        inline std::string get_environment_value(const std::string &env_name);

        /// This function detects an equal or colon followed by an escaped quote after an argument
        /// then modifies the string to replace the equality with a space.  This is needed
        /// to allow the split up function to work properly and is intended to be used with the find_and_modify function
        /// the return value is the offset+1 which is required by the find_and_modify function.
        inline std::size_t escape_detect(std::string &str, std::size_t offset);

        /// @brief  detect if a string has escapable characters
        /// @param str the string to do the detection on
        /// @return true if the string has escapable characters
        inline bool has_escapable_character(const std::string &str);

        /// @brief escape all escapable characters
        /// @param str the string to escape
        /// @return a string with the escapble characters escaped with '\'
        inline std::string add_escaped_characters(const std::string &str);

        /// @brief replace the escaped characters with their equivalent
        inline std::string remove_escaped_characters(const std::string &str);

        /// generate a string with all non printable characters escaped to hex codes
        inline std::string binary_escape_string(const std::string &string_to_escape);

        inline bool is_binary_escaped_string(const std::string &escaped_string);

        /// extract an escaped binary_string
        inline std::string extract_binary_string(const std::string &escaped_string);

        /// process a quoted string, remove the quotes and if appropriate handle escaped characters
        inline bool process_quoted_string(std::string &str, char string_char = '\"', char literal_char = '\'');

    }  // namespace detail


}  // namespace collie

#include <collie/cli/impl/string_tools_inl.h>
