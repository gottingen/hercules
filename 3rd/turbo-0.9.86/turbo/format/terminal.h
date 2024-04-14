// Copyright 2023 The Elastic-AI Authors.
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


#ifndef TURBO_FORMAT_TERMINAL_H_
#define TURBO_FORMAT_TERMINAL_H_

#include "turbo/platform/port.h"
#include "turbo/format/color.h"
#include "turbo/format/format.h"
#include <ostream>
#include <cstdio>
#include <iostream>


// This headers provides the `isatty()`/`fileno()` functions,
// which are used for testing whether a standart stream refers
// to the terminal. As for Windows, we also need WinApi funcs
// for changing colors attributes of the terminal.
#if defined(TURBO_PLATFORM_OSX) || defined(TURBO_PLATFORM_LINUX)

#include <unistd.h>

#elif defined(TURBO_PLATFORM_WINDOWS)
#include <io.h>
#include <windows.h>
#endif

namespace turbo::fmt_detail {
    // An index to be used to access a private storage of I/O streams. See
    // colorize / nocolorize I/O manipulators for details.
    static int colorize_index = std::ios_base::xalloc();

    inline FILE *get_standard_stream(const std::ostream &stream);

    inline bool is_colorized(std::ostream &stream);

    inline bool is_atty(const std::ostream &stream);

} // namespace turbo::fmt_detail

namespace turbo {
    inline std::ostream &colorize(std::ostream &stream) {
        stream.iword(fmt_detail::colorize_index) = 1L;
        return stream;
    }

    inline std::ostream &nocolorize(std::ostream &stream) {
        stream.iword(fmt_detail::colorize_index) = 0L;
        return stream;
    }

    inline std::ostream &reset_style(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            stream << turbo::reset_text_style();
        }
        return stream;
    }

    inline std::ostream &apply_style(std::ostream &stream, const text_style &style) {
        if (fmt_detail::is_colorized(stream)) {
            stream << apply_text_style(style);
        }
        return stream;
    }

    inline std::ostream &apply_style(std::ostream &stream, const text_style_builder &style) {
        if (fmt_detail::is_colorized(stream)) {
            stream << apply_text_style(style);
        }
        return stream;
    }

    static constexpr std::string_view kBackLine = "\033[F";

    inline std::string backwards_lines(size_t n) {
        std::string result;
        result.reserve(n * kBackLine.size());
        for (size_t i = 0; i < n; ++i) {
            result.append(kBackLine.data(), kBackLine.size());
        }
        return result;
    }

    inline std::ostream &backwards_lines(std::ostream &stream, size_t n) {
        if (fmt_detail::is_colorized(stream)) {
            stream << backwards_lines(n);
        }
        return stream;
    }

    static constexpr std::string_view kForwardLine = "\033[B";

    inline std::string forward_lines(size_t n) {
        std::string result;
        result.reserve(n * kForwardLine.size());
        for (size_t i = 0; i < n; ++i) {
            result.append(kForwardLine.data(), kForwardLine.size());
        }
        return result;
    }

    inline std::ostream &forward_lines(std::ostream &stream, size_t n) {
        if (fmt_detail::is_colorized(stream)) {
            stream << forward_lines(n);
        }
        return stream;
    }

    inline std::ostream &bold(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bold();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &dark(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.faint();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &italic(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.italic();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &underline(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.underline();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &blink(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.blink();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &reverse(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.reverse();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &concealed(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.conceal();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &crossed(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.strikethrough();
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &black(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::black);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &red(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::red);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &green(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::green);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &yellow(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::yellow);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &blue(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::blue);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &magenta(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::magenta);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &cyan(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::cyan);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &white(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.fg(terminal_color::white);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_black(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::black);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_red(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::red);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_green(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::green);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_yellow(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::yellow);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_blue(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::blue);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_magenta(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::magenta);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_cyan(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::cyan);
            stream << builder.build();
        }
        return stream;
    }

    inline std::ostream &on_white(std::ostream &stream) {
        if (fmt_detail::is_colorized(stream)) {
            text_style_builder builder;
            builder.bg(terminal_color::white);
            stream << builder.build();
        }
        return stream;
    }
}  // namespace turbo

namespace turbo::fmt_detail {
    //! Since C++ hasn't a true way to extract stream handler
    //! from the a given `std::ostream` object, I have to write
    //! this kind of hack.
    inline FILE *get_standard_stream(const std::ostream &stream) {
        if (&stream == &std::cout)
            return stdout;
        else if ((&stream == &std::cerr) || (&stream == &std::clog))
            return stderr;

        return 0;
    }

    // Say whether a given stream should be colorized or not. It's always
    // true for ATTY streams and may be true for streams marked with
    // colorize flag.
    inline bool is_colorized(std::ostream &stream) {
        return is_atty(stream) || static_cast<bool>(stream.iword(colorize_index));
    }

    //! Test whether a given `std::ostream` object refers to
    //! a terminal.
    inline bool is_atty(const std::ostream &stream) {
        FILE *std_stream = get_standard_stream(stream);

        // Unfortunately, fileno() ends with segmentation fault
        // if invalid file descriptor is passed. So we need to
        // handle this case gracefully and assume it's not a tty
        // if standard stream is not detected, and 0 is returned.
        if (!std_stream)
            return false;

        return ::isatty(fileno(std_stream));
    }
} // namespace turbo::fmt_detail

#endif  // TURBO_FORMAT_TERMINAL_H_
