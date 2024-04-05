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
#pragma once

#include <cctype>
#include <collie/log/common.h>
#include <collie/strings/fmt/format.h>

#if defined(__has_include)
#if __has_include(<version>)

#include <version>

#endif
#endif

#if __cpp_lib_span >= 202002L
#include <span>
#endif

//
// Support for logging binary data as hex
// format flags, any combination of the following:
// {:X} - print in uppercase.
// {:s} - don't separate each byte with space.
// {:p} - don't print the position on each line start.
// {:n} - don't split the output to lines.
// {:a} - show ASCII if :n is not set

//
// Examples:
//
// std::vector<char> v(200, 0x0b);
// logger->info("Some buffer {}", clog::to_hex(v));
// char buf[128];
// logger->info("Some buffer {:X}", clog::to_hex(std::begin(buf), std::end(buf)));
// logger->info("Some buffer {:X}", clog::to_hex(std::begin(buf), std::end(buf), 16));

namespace clog {
    namespace details {

        template<typename It>
        class dump_info {
        public:
            dump_info(It range_begin, It range_end, size_t size_per_line)
                    : begin_(range_begin),
                      end_(range_end),
                      size_per_line_(size_per_line) {}

            // do not use begin() and end() to avoid collision with fmt/ranges
            It get_begin() const { return begin_; }

            It get_end() const { return end_; }

            size_t size_per_line() const { return size_per_line_; }

        private:
            It begin_, end_;
            size_t size_per_line_;
        };
    }  // namespace details

    // create a dump_info that wraps the given container
    template<typename Container>
    inline details::dump_info<typename Container::const_iterator> to_hex(const Container &container,
                                                                         size_t size_per_line = 32) {
        static_assert(sizeof(typename Container::value_type) == 1,
                      "sizeof(Container::value_type) != 1");
        using Iter = typename Container::const_iterator;
        return details::dump_info<Iter>(std::begin(container), std::end(container), size_per_line);
    }

#if __cpp_lib_span >= 202002L

    template <typename Value, size_t Extent>
    inline details::dump_info<typename std::span<Value, Extent>::iterator> to_hex(
        const std::span<Value, Extent> &container, size_t size_per_line = 32) {
        using Container = std::span<Value, Extent>;
        static_assert(sizeof(typename Container::value_type) == 1,
                      "sizeof(Container::value_type) != 1");
        using Iter = typename Container::iterator;
        return details::dump_info<Iter>(std::begin(container), std::end(container), size_per_line);
    }

#endif

    // create dump_info from ranges
    template<typename It>
    inline details::dump_info<It> to_hex(const It range_begin,
                                         const It range_end,
                                         size_t size_per_line = 32) {
        return details::dump_info<It>(range_begin, range_end, size_per_line);
    }

}  // namespace clog

namespace fmt {

    template<typename T>
    struct formatter<clog::details::dump_info<T>, char> {
        const char delimiter = ' ';
        bool put_newlines = true;
        bool put_delimiters = true;
        bool use_uppercase = false;
        bool put_positions = true;  // position on start of each line
        bool show_ascii = false;

        // parse the format string flags
        template<typename ParseContext>
        constexpr auto parse(ParseContext &ctx) -> decltype(ctx.begin()) {
            auto it = ctx.begin();
            while (it != ctx.end() && *it != '}') {
                switch (*it) {
                    case 'X':
                        use_uppercase = true;
                        break;
                    case 's':
                        put_delimiters = false;
                        break;
                    case 'p':
                        put_positions = false;
                        break;
                    case 'n':
                        put_newlines = false;
                        show_ascii = false;
                        break;
                    case 'a':
                        if (put_newlines) {
                            show_ascii = true;
                        }
                        break;
                }

                ++it;
            }
            return it;
        }

        // format the given bytes range as hex
        template<typename FormatContext, typename Container>
        auto format(const clog::details::dump_info<Container> &the_range, FormatContext &ctx) const
        -> decltype(ctx.out()) {
            constexpr const char *hex_upper = "0123456789ABCDEF";
            constexpr const char *hex_lower = "0123456789abcdef";
            const char *hex_chars = use_uppercase ? hex_upper : hex_lower;
            auto inserter = ctx.out();

            int size_per_line = static_cast<int>(the_range.size_per_line());
            auto start_of_line = the_range.get_begin();
            for (auto i = the_range.get_begin(); i != the_range.get_end(); i++) {
                auto ch = static_cast<unsigned char>(*i);

                if (put_newlines &&
                    (i == the_range.get_begin() || i - start_of_line >= size_per_line)) {
                    if (show_ascii && i != the_range.get_begin()) {
                        *inserter++ = delimiter;
                        *inserter++ = delimiter;
                        for (auto j = start_of_line; j < i; j++) {
                            auto pc = static_cast<unsigned char>(*j);
                            *inserter++ = std::isprint(pc) ? static_cast<char>(*j) : '.';
                        }
                    }

                    put_newline(inserter, static_cast<size_t>(i - the_range.get_begin()));

                    // put first byte without delimiter in front of it
                    *inserter++ = hex_chars[(ch >> 4) & 0x0f];
                    *inserter++ = hex_chars[ch & 0x0f];
                    start_of_line = i;
                    continue;
                }

                if (put_delimiters && i != the_range.get_begin()) {
                    *inserter++ = delimiter;
                }

                *inserter++ = hex_chars[(ch >> 4) & 0x0f];
                *inserter++ = hex_chars[ch & 0x0f];
            }
            if (show_ascii)  // add ascii to last line
            {
                if (the_range.get_end() - the_range.get_begin() > size_per_line) {
                    auto blank_num = size_per_line - (the_range.get_end() - start_of_line);
                    while (blank_num-- > 0) {
                        *inserter++ = delimiter;
                        *inserter++ = delimiter;
                        if (put_delimiters) {
                            *inserter++ = delimiter;
                        }
                    }
                }
                *inserter++ = delimiter;
                *inserter++ = delimiter;
                for (auto j = start_of_line; j != the_range.get_end(); j++) {
                    auto pc = static_cast<unsigned char>(*j);
                    *inserter++ = std::isprint(pc) ? static_cast<char>(*j) : '.';
                }
            }
            return inserter;
        }

        // put newline(and position header)
        template<typename It>
        void put_newline(It inserter, std::size_t pos) const {
#ifdef _WIN32
            *inserter++ = '\r';
#endif
            *inserter++ = '\n';

            if (put_positions) {
                clog::fmt_lib::format_to(inserter, FMT_STRING("{:04X}: "), pos);
            }
        }
    };
}  // namespace std
