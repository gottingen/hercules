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

#include <chrono>
#include <iterator>
#include <collie/log/common.h>
#include <collie/strings/fmt/format.h>
#include <type_traits>


// Some fmt helpers to efficiently format and pad ints and strings
namespace clog {
namespace details {
namespace fmt_helper {

inline void append_string_view(clog::string_view_t view, memory_buf_t &dest) {
    auto *buf_ptr = view.data();
    dest.append(buf_ptr, buf_ptr + view.size());
}


template <typename T>
inline void append_int(T n, memory_buf_t &dest) {
    fmt::format_int i(n);
    dest.append(i.data(), i.data() + i.size());
}

template <typename T>
constexpr unsigned int count_digits_fallback(T n) {
    // taken from fmt: https://github.com/fmtlib/fmt/blob/8.0.1/include/fmt/format.h#L899-L912
    unsigned int count = 1;
    for (;;) {
        // Integer division is slow so do it for a group of four digits instead
        // of for every digit. The idea comes from the talk by Alexandrescu
        // "Three Optimization Tips for C++". See speed-test for a comparison.
        if (n < 10) return count;
        if (n < 100) return count + 1;
        if (n < 1000) return count + 2;
        if (n < 10000) return count + 3;
        n /= 10000u;
        count += 4;
    }
}

template <typename T>
inline unsigned int count_digits(T n) {
    using count_type =
        typename std::conditional<(sizeof(T) > sizeof(uint32_t)), uint64_t, uint32_t>::type;
    return static_cast<unsigned int>(fmt::detail::count_digits(static_cast<count_type>(n)));
}

inline void pad2(int n, memory_buf_t &dest) {
    if (n >= 0 && n < 100)  // 0-99
    {
        dest.push_back(static_cast<char>('0' + n / 10));
        dest.push_back(static_cast<char>('0' + n % 10));
    } else  // unlikely, but just in case, let fmt deal with it
    {
        fmt_lib::format_to(std::back_inserter(dest), FMT_STRING("{:02}"), n);
    }
}

template <typename T>
inline void pad_uint(T n, unsigned int width, memory_buf_t &dest) {
    static_assert(std::is_unsigned<T>::value, "pad_uint must get unsigned T");
    for (auto digits = count_digits(n); digits < width; digits++) {
        dest.push_back('0');
    }
    append_int(n, dest);
}

template <typename T>
inline void pad3(T n, memory_buf_t &dest) {
    static_assert(std::is_unsigned<T>::value, "pad3 must get unsigned T");
    if (n < 1000) {
        dest.push_back(static_cast<char>(n / 100 + '0'));
        n = n % 100;
        dest.push_back(static_cast<char>((n / 10) + '0'));
        dest.push_back(static_cast<char>((n % 10) + '0'));
    } else {
        append_int(n, dest);
    }
}

template <typename T>
inline void pad6(T n, memory_buf_t &dest) {
    pad_uint(n, 6, dest);
}

template <typename T>
inline void pad9(T n, memory_buf_t &dest) {
    pad_uint(n, 9, dest);
}

// return fraction of a second of the given time_point.
// e.g.
// fraction<std::milliseconds>(tp) -> will return the millis part of the second
template <typename ToDuration>
inline ToDuration time_fraction(log_clock::time_point tp) {
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    auto duration = tp.time_since_epoch();
    auto secs = duration_cast<seconds>(duration);
    return duration_cast<ToDuration>(duration) - duration_cast<ToDuration>(secs);
}

}  // namespace fmt_helper
}  // namespace details
}  // namespace clog
