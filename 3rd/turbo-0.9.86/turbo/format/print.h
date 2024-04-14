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

#ifndef TURBO_FORMAT_PRINT_H_
#define TURBO_FORMAT_PRINT_H_

#include <cstdint>
#include "turbo/format/format.h"
#include "turbo/format/fmt/printf.h"
#include "turbo/format/fmt/os.h"
#include "turbo/format/fmt/ostream.h"
#include "turbo/format/color.h"

namespace turbo {


    /**
      * @ingroup turbo_fmt_print
      * @brief Formats ``args`` according to specifications in ``fmt`` and writes the output to ``stdout``.
      *         Example:
      *         @code{.cpp}
      *         turbo::print("Elapsed time: {0:.2f} seconds", 1.23);
      *         @endcode
      *         Output:
      *         @code{.unparsed}
      *         Elapsed time: 1.23 seconds
      *         @endcode
      *         The print module is part of the turbo::format module.
      */
    template<typename... T>
    TURBO_FORCE_INLINE void print(format_string<T...> fmt, T &&... args) {
        const auto &vargs = turbo::make_format_args(args...);
        return fmt_detail::is_utf8() ? vprint(fmt, vargs)
                                 : fmt_detail::vprint_mojibake(stdout, fmt, vargs);
    }

    /**
      \rst
      Formats ``args`` according to specifications in ``fmt`` and writes the
      output to the file ``f``.

      **Example**::

        turbo::print(stderr, "Don't {}!", "panic");
      \endrst
     */
    template<typename... T>
    TURBO_FORCE_INLINE void print(std::FILE *f, format_string<T...> fmt, T &&... args) {
        const auto &vargs = turbo::make_format_args(args...);
        return fmt_detail::is_utf8() ? vprint(f, fmt, vargs)
                                 : fmt_detail::vprint_mojibake(f, fmt, vargs);
    }


    /**
      \rst
      Prints formatted data to the stream *os*.

      **Example**::

        turbo::print(cerr, "Don't {}!", "panic");
      \endrst
     */
    template<typename... T>
    void print(std::ostream &os, format_string<T...> fmt, T &&... args) {
        const auto &vargs = turbo::make_format_args(args...);
        if (fmt_detail::is_utf8())
            vprint(os, fmt, vargs);
        else
            fmt_detail::vprint_directly(os, fmt, vargs);
    }


    template<typename... Args>
    void print(std::wostream &os,
               basic_format_string<wchar_t, type_identity_t<Args>...> fmt,
               Args &&... args) {
        vprint(os, fmt, turbo::make_format_args<buffer_context<wchar_t>>(args...));
    }

    /**
      \rst
      Formats a string and prints it to the specified file stream using ANSI
      escape sequences to specify text formatting.

      **Example**::

        turbo::print(turbo::emphasis::bold | fg(turbo::color::red),
                   "Elapsed time: {0:.2f} seconds", 1.23);
      \endrst
     */
    template<typename S, typename... Args,
            TURBO_ENABLE_IF(is_string<S>::value)>

    void print(std::FILE *f, const text_style &ts, const S &format_str,
               const Args &... args) {
        vprint(f, ts, format_str,
               turbo::make_format_args<buffer_context<char_t<S>>>(args...));
    }

    /**
      \rst
      Formats a string and prints it to stdout using ANSI escape sequences to
      specify text formatting.

      **Example**::

        turbo::print(turbo::emphasis::bold | fg(turbo::color::red),
                   "Elapsed time: {0:.2f} seconds", 1.23);
      \endrst
     */
    template<typename S, typename... Args,
            TURBO_ENABLE_IF(is_string<S>::value)>

    void print(const text_style &ts, const S &format_str, const Args &... args) {
        return print(stdout, ts, format_str, args...);
    }


    /**
          Formats ``args`` according to specifications in ``fmt`` and writes the
          output to the file ``f`` followed by a newline.
         */
    template<typename... T>
    TURBO_FORCE_INLINE void println(std::FILE *f, format_string<T...> fmt, T &&... args) {
        return turbo::print(f, "{}\n", turbo::format(fmt, std::forward<T>(args)...));
    }

    /**
          Formats ``args`` according to specifications in ``fmt`` and writes the output
          to ``stdout`` followed by a newline.
         */
    template<typename... T>
    TURBO_FORCE_INLINE void println(format_string<T...> fmt, T &&... args) {
        return turbo::println(stdout, fmt, std::forward<T>(args)...);
    }

    template<typename... T>
    void println(std::ostream &os, format_string<T...> fmt, T &&... args) {
        turbo::print(os, "{}\n", turbo::format(fmt, std::forward<T>(args)...));
    }


    template<typename... Args>
    void println(std::wostream &os,
                 basic_format_string<wchar_t, type_identity_t<Args>...> fmt,
                 Args &&... args) {
        print(os, L"{}\n", turbo::format(fmt, std::forward<Args>(args)...));
    }



    /**
      \rst
      Prints formatted data to the file *f*.

      **Example**::

        turbo::fprintf(stderr, "Don't %s!", "panic");
      \endrst
     */
    template<typename S, typename... T, typename Char = char_t<S>>
    inline auto fprintf(std::FILE *f, const S &fmt, const T &... args) -> int {
        using context = basic_printf_context_t<Char>;
        return vfprintf(f, to_string_view(fmt),
                        turbo::make_format_args<context>(args...));
    }

    /**
      \rst
      Formats arguments and returns the result as a string.

      **Example**::

        std::string message = turbo::sprintf("The answer is %d", 42);
      \endrst
    */
    template<typename S, typename... T,
            typename Char = std::enable_if_t<is_string<S>::value, char_t<S>>>
    inline auto sprintf(const S &fmt, const T &... args) -> std::basic_string<Char> {
        using context = basic_printf_context_t<Char>;
        return vsprintf(to_string_view(fmt),
                        turbo::make_format_args<context>(args...));
    }

    /**
      \rst
      Prints formatted data to ``stdout``.

      **Example**::

        turbo::printf("Elapsed time: %.2f seconds", 1.23);
      \endrst
     */
    template<typename S, typename... T, TURBO_ENABLE_IF(is_string<S>::value)>
    inline auto printf(const S &fmt, const T &... args) -> int {
        return vprintf(to_string_view(fmt),
                turbo::make_format_args<basic_printf_context_t<char_t<S>>>(args...));
    }

    template<typename ...Args>
    void Print(std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, fmt, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void Println(std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    void FPrint(std::FILE *file, std::string_view fmt,  Args &&... args) {
        turbo::print(file, fmt, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void FPrintln(std::FILE *file, std::string_view fmt,  Args &&... args) {
        turbo::print(file, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    using turbo::color;
    using turbo::bg;
    using turbo::fg;
    using turbo::text_style;

    static const text_style RedFG = fg(color::red);
    static const text_style GreenFG = fg(color::green);
    static const text_style YellowFG = fg(color::yellow);

    template<typename ...Args>
    void Print(const text_style& ts, std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, ts, fmt, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void Println(const text_style& ts, std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, ts, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    void Print(const color& c, std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, fg(c), fmt, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    void Println(const color& c, std::string_view fmt,  Args &&... args) {
        turbo::print(stdout, fg(c), "{}\n", format(fmt, std::forward<Args>(args)...));
    }

}  // namespace turbo

#endif  // TURBO_FORMAT_PRINT_H_
