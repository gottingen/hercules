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

#ifndef TURBO_STRING_FIXED_STRING_H_
#define TURBO_STRING_FIXED_STRING_H_

#include <string>
#include <cstring>
#include <algorithm>
#include <array>
#include <type_traits>
#include <string_view>
#include "turbo/platform/port.h"
#include "turbo/format/format.h"
#include <ostream>

namespace turbo {

    template<class CharT, std::size_t max_length, class Traits = std::char_traits<CharT>>
    class basic_str {
    public:
        constexpr basic_str() noexcept = default;

        template<std::size_t N>
        basic_str(const char (&str)[N]) : active_length_(std::min(N - 1, max_length)) {
            assert(str[N - 1] == '\0');
            Traits::copy(buffer_, str, active_length_);
            buffer_[active_length_] = '\0';
        }

        basic_str(const CharT *str)
                : active_length_(std::min(Traits::length(str), max_length)) {
            Traits::copy(buffer_, str, active_length_);
        }

        basic_str(const CharT *str, std::size_t length)
                : active_length_(std::min(length, max_length)) {
            Traits::copy(buffer_, str, active_length_);
        }

        constexpr const CharT *c_str() const noexcept { return buffer_; }

        constexpr const CharT *data() const noexcept { return buffer_; }

        constexpr CharT *data() noexcept { return buffer_; }

        constexpr std::basic_string_view<CharT, Traits> str() const noexcept {
            return std::basic_string_view<CharT, Traits>(buffer_, active_length_);
        }

        constexpr auto length() const noexcept { return active_length_; }

        constexpr auto max_size() const noexcept { return max_length; }

        constexpr auto empty() const noexcept { return active_length_ == 0; }

        constexpr void clear() noexcept {
            active_length_ = 0;
            buffer_[0] = '\0';
        }

        constexpr void reset(const CharT *str) {
            active_length_ = std::min(Traits::length(str), max_length);
            reset_(str, active_length_);
        }

        constexpr void reset(const CharT *str, std::size_t length) {
            active_length_ = std::min(length, max_length);
            reset_(str, active_length_);
        }

        constexpr void append(const CharT *str) {
            auto to_copy = std::min(Traits::length(str), (max_length - active_length_));
            append_(str, to_copy);
        }

        constexpr void append(const CharT *str, std::size_t length) {
            auto to_copy = std::min(length, (max_length - active_length_));
            append_(str, to_copy);
        }

        constexpr void append(const std::basic_string_view<CharT, Traits> &str) {
            auto to_copy = std::min(str.length(), (max_length - active_length_));
            append_(str.data(), to_copy);
        }

        constexpr void append(size_t n, CharT c) {
            auto to_copy = std::min(n, (max_length - active_length_));
            std::fill_n(buffer_ + active_length_, to_copy, c);
            active_length_ += to_copy;
            buffer_[active_length_] = '\0';
        }

        constexpr void append(const CharT *first, const CharT *last) {
            auto to_copy = std::min(static_cast<std::size_t>(last - first), (max_length - active_length_));
            append_(first, to_copy);
        }

        constexpr void remove_prefix(std::size_t length) {
            std::copy(buffer_ + length, buffer_ + active_length_, buffer_);
            active_length_ -= length;
            buffer_[active_length_] = '\0';
        }

        constexpr void remove_suffix(std::size_t length) noexcept {
            active_length_ = active_length_ - length;
            buffer_[active_length_] = '\0';
        }

        /* implemented as a member to avoid implicit conversion */
        constexpr bool operator==(const basic_str &rhs) const {
            return (max_size() == rhs.max_size()) && (length() == rhs.length()) &&
                   std::equal(buffer_, buffer_ + length(), rhs.buffer_);
        }

        constexpr bool operator!=(const basic_str &rhs) const {
            return !(*this == rhs);
        }

        constexpr void swap(basic_str &rhs) noexcept {
            std::swap(active_length_, rhs.active_length_);
            std::swap(buffer_, rhs.buffer_);
        }

        constexpr operator std::basic_string_view<CharT, Traits>() const noexcept {
            return str();
        }

    private:
        constexpr void reset_(const CharT *str, std::size_t length) {
            Traits::copy(buffer_, str, length);
            buffer_[length] = '\0';
        }

        constexpr void append_(const CharT *str, std::size_t to_copy) {
            std::copy(str, str + to_copy, buffer_ + active_length_);
            active_length_ += to_copy;
            buffer_[active_length_] = '\0';
        }

        std::size_t active_length_{0};
        CharT buffer_[max_length + 1]{};
    };

    template<class CharT, std::size_t max_length, class Traits = std::char_traits<CharT>>
    inline constexpr void
    swap(const basic_str<CharT, max_length> &lhs, const basic_str<CharT, max_length> &rhs) noexcept {
        rhs.swap(lhs);
    }

    template<class CharT, std::size_t max_length, class Traits = std::char_traits<CharT>>
    inline std::basic_ostream<CharT, Traits> &
    operator<<(std::basic_ostream<CharT, Traits> &os, const basic_str<CharT, max_length> &str) {
        return os << str.c_str();
    }

    template<typename CharT, std::size_t max_length, typename Traits>
    struct formatter<basic_str<CharT, max_length,Traits>,CharT>
            : formatter<std::basic_string_view<CharT, Traits>, CharT> {
        template<typename FormatContext>
        auto format(const basic_str<CharT, max_length> &str, FormatContext &ctx) {
            return formatter<std::basic_string_view<CharT, Traits>, CharT>::format(str.str(), ctx);
        }
    };

    template<std::size_t max_length>
    using fixed_string = basic_str<char, max_length>;

    template<std::size_t max_length>
    using fixed_wstring = basic_str<wchar_t, max_length>;

    template<std::size_t max_length>
    using fixed_u8string = basic_str<char, max_length>;

    template<std::size_t max_length>
    using fixed_u16string = basic_str<char16_t, max_length>;

    template<std::size_t max_length>
    using fixed_32string = basic_str<char32_t, max_length>;
}
#endif // TURBO_STRING_FIXED_STRING_H_
