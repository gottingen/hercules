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

#ifndef TURBO_FORMAT_FORMAT_H_
#define TURBO_FORMAT_FORMAT_H_

#include <string>
#include "turbo/format/fmt/format.h"
#include "turbo/format/fmt/core.h"
#include "turbo/format/fmt/ranges.h"
#include "turbo/format/fmt/printf.h"
#include "turbo/format/fmt/std.h"
#include "turbo/format/color.h"
#include "turbo/platform/port.h"

namespace turbo {

    /**
     * @defgroup turbo_fmt_format turbo::format
     * @brief The format module provides a fast and safe alternative to C stdio and C++ iostreams.
     * @details The format module provides a fast and safe alternative to C stdio and C++ iostreams.
     *          It is designed to be easy to use and to handle large amounts of data.
     *          The format module is part of the turbo::format module.
     *
     *          Example:
     *          @code{.cpp}
     *          std::string message = turbo::format("The answer is {}.", 42);
     *          @endcode
     *          Output:
     *          @code{.unparsed}
     *          The answer is 42.
     *          @endcode
     * @ingroup turbo_fmt_format
     */
    template<typename... T>
    [[nodiscard]] TURBO_FORCE_INLINE auto format(format_string<T...> fmt, T &&... args) -> std::string {
        return vformat(fmt, turbo::make_format_args(args...));
    }

    template<typename Locale, typename... T,
            TURBO_ENABLE_IF(fmt_detail::is_locale<Locale>::value)>
    inline auto format(const Locale &loc, format_string<T...> fmt, T &&... args)
    -> std::string {
        return turbo::vformat(loc, std::string_view(fmt), turbo::make_format_args(args...));
    }

    /**
     \rst
     Returns an argument that will be formatted using ANSI escape sequences,
     to be used in a formatting function.

     **Example**::

       turbo::print("Elapsed time: {0:.2f} seconds",
                  turbo::styled(1.23, turbo::fg(turbo::color::green) |
                                    turbo::bg(turbo::color::blue)));
     \endrst
    */
    template<typename T>
    constexpr auto styled(const T &value, text_style ts) -> fmt_detail::styled_arg<turbo::remove_cvref_t<T>> {
        return fmt_detail::styled_arg<turbo::remove_cvref_t<T>>
                {value, ts};
    }

    /**
      \rst
      Formats arguments and returns the result as a string using ANSI
      escape sequences to specify text formatting.

      **Example**::

        #include <fmt/color.h>
        std::string message = turbo::format(turbo::emphasis::bold | fg(turbo::color::red),
                                          "The answer is {}", 42);
      \endrst
    */
    template<typename S, typename... Args, typename Char = char_t<S>>
    inline std::basic_string<Char> format(const text_style &ts, const S &format_str,
                                          const Args &... args) {
        return turbo::vformat(ts, to_string_view(format_str),
                              turbo::make_format_args<buffer_context<Char>>(args...));
    }

    /**
     \rst
     Formats arguments with the given text_style, writes the result to the output
     iterator ``out`` and returns the iterator past the end of the output range.

     **Example**::

       std::vector<char> out;
       turbo::format_to(std::back_inserter(out),
                      turbo::emphasis::bold | fg(turbo::color::red), "{}", 42);
     \endrst
   */
    template<typename OutputIt, typename S, typename... Args,
            bool enable = fmt_detail::is_output_iterator<OutputIt, char_t<S>>::value &&
                          is_string<S>::value
    >

    inline auto format_to(OutputIt out, const text_style &ts, const S &format_str,
                          Args &&... args) ->
    typename std::enable_if<enable, OutputIt>::type {
        return vformat_to(out, ts, to_string_view(format_str),
                          turbo::make_format_args<buffer_context<char_t<S>>>(args...));
    }


    template<typename String = std::string, typename T>
    TURBO_MUST_USE_RESULT inline String format(const T &t) {
        String result;
        turbo::memory_buffer buf;
        turbo::format_to(std::back_inserter(buf), "{}", t);
        return String(buf.data(), buf.size());
    }

    template<typename String = std::string, typename ...Args>
    void format_append(String *dst, std::string_view fmt, Args &&... args) {
        turbo::memory_buffer buf;
        turbo::format_to(std::back_inserter(buf), fmt, std::forward<Args>(args)...);
        dst->append(buf.data(), buf.size());
    }

    template<typename String = std::string, typename T>
    void format_append(String *dst, const T &t) {
        turbo::memory_buffer buf;
        turbo::format_to(std::back_inserter(buf), "{}", t);
        dst->append(buf.data(), buf.size());
    }


    template<typename String = std::string, typename ...Args>
    String format_range(std::string_view fmt, const std::tuple<Args...> &tuple, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(tuple, sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename T>

    String format_range(std::string_view fmt, std::initializer_list<T> list, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(list, sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename It, typename Sentinel, typename String = std::string>
    String format_range(std::string_view fmt, It begin, Sentinel end, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt,
                         turbo::join(std::forward<It>(begin), std::forward<Sentinel>(end), sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename Range>
    String format_range(std::string_view fmt, Range &&range, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(std::forward<Range>(range), sep));
        return String(view_buf.data(), view_buf.size());
    }

    /// format_range_append
    template<typename String = std::string, typename ...Args>
    void
    format_range_append(String *dst, std::string_view fmt, const std::tuple<Args...> &tuple, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(tuple, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename T>

    void format_range_append(String *dst, std::string_view fmt, std::initializer_list<T> list, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(list, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename It, typename Sentinel>
    void format_range_append(String *dst, std::string_view fmt, It begin, Sentinel end, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(begin, end, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename Range>
    void format_range_append(String *dst, std::string_view fmt, Range &&range, std::string_view sep) {
        turbo::memory_buffer view_buf;
        turbo::format_to(std::back_inserter(view_buf), fmt, turbo::join(std::forward<Range>(range), sep));
        dst->append(view_buf.data(), view_buf.size());
    }


}  // namespace turbo

#endif  // TURBO_FORMAT_FORMAT_H_
