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

#ifndef COLLIE_STRINGS_FORMAT_H_
#define COLLIE_STRINGS_FORMAT_H_

#include <collie/strings/fmt/format.h>
#include <collie/strings/fmt/ostream.h>
#include <collie/strings/fmt/printf.h>
#include <collie/strings/fmt/ranges.h>
#include <collie/strings/fmt/chrono.h>
#include <collie/strings/fmt/color.h>
#include <collie/strings/fmt/compile.h>
#include <collie/strings/fmt/os.h>
#include <collie/strings/fmt/args.h>
#include <collie/strings/fmt/std.h>

namespace collie {

    template<typename String = std::string, typename T>
    [[nodiscard]] inline String to_str(const T &t) {
        String result;
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf), "{}", t);
        return String(buf.data(), buf.size());
    }

    using namespace fmt;

    template<typename String = std::string, typename ...Args>
    void format_append(String *dst, std::string_view fmt, Args &&... args) {
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf), fmt, std::forward<Args>(args)...);
        dst->append(buf.data(), buf.size());
    }

    template<typename String = std::string, typename T>
    void format_append(String *dst, const T &t) {
        fmt::memory_buffer buf;
        fmt::format_to(std::back_inserter(buf), "{}", t);
        dst->append(buf.data(), buf.size());
    }

    template<typename String = std::string, typename ...Args>
    String format_range(std::string_view fmt, const std::tuple<Args...> &tuple, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(tuple, sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename T>

    String format_range(std::string_view fmt, std::initializer_list<T> list, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(list, sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename It, typename Sentinel, typename String = std::string>
    String format_range(std::string_view fmt, It begin, Sentinel end, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt,
                         fmt::join(std::forward<It>(begin), std::forward<Sentinel>(end), sep));
        return String(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename Range>
    String format_range(std::string_view fmt, Range &&range, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(std::forward<Range>(range), sep));
        return String(view_buf.data(), view_buf.size());
    }

    /// format_range_append
    template<typename String = std::string, typename ...Args>
    void
    format_range_append(String *dst, std::string_view fmt, const std::tuple<Args...> &tuple, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(tuple, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename T>

    void format_range_append(String *dst, std::string_view fmt, std::initializer_list<T> list, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(list, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename It, typename Sentinel>
    void format_range_append(String *dst, std::string_view fmt, It begin, Sentinel end, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(begin, end, sep));
        dst->append(view_buf.data(), view_buf.size());
    }

    template<typename String = std::string, typename Range>
    void format_range_append(String *dst, std::string_view fmt, Range &&range, std::string_view sep) {
        fmt::memory_buffer view_buf;
        fmt::format_to(std::back_inserter(view_buf), fmt, fmt::join(std::forward<Range>(range), sep));
        dst->append(view_buf.data(), view_buf.size());
    }


    template<typename ...Args>
    void println(std::FILE *file, std::string_view fmt, Args &&... args) {
        fmt::print(file, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    void println(std::ostream &os, std::string_view fmt, Args &&... args) {
        fmt::print(os, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    inline void println(std::string_view fmt, Args &&... args) {
        fmt::print(stdout, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    inline void println(const text_style &ts, std::string_view fmt, Args &&... args) {
        fmt::print(stdout, ts, "{}\n", format(fmt, std::forward<Args>(args)...));
    }

    template<typename ...Args>
    inline void println(const color &c, std::string_view fmt, Args &&... args) {
        fmt::print(stdout, fg(c), "{}\n", format(fmt, std::forward<Args>(args)...));
    }


}

#endif  // COLLIE_STRINGS_FORMAT_H_
