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

#ifndef TURBO_MODULE_MODULE_VERSION_H_
#define TURBO_MODULE_MODULE_VERSION_H_

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include "turbo/format/format.h"
#include "turbo/strings/ascii.h"
#include "turbo/strings/match.h"

#if __has_include(<charconv>)

#include <charconv>

#else
#include <system_error>
#endif

#if defined(SEMVER_CONFIG_FILE)
#include SEMVER_CONFIG_FILE
#endif

#if defined(TURBO_MODULE_THROW)
// define TURBO_MODULE_THROW(msg) to override turbo throw behavior.
#elif defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)

#  include <stdexcept>

#  define TURBO_MODULE_THROW(msg) (throw std::invalid_argument{msg})
#else
#  include <cassert>
#  include <cstdlib>
#  define TURBO_MODULE_THROW(msg) (assert(!msg), std::abort())
#endif

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wmissing-braces" // Ignore warning: suggest braces around initialization of subobject 'return {first, std::errc::invalid_argument};'.
#endif

namespace turbo {

    enum struct prerelease : std::uint8_t {
        alpha = 0,
        beta = 1,
        rc = 2,
        none = 3
    };

#if __has_include(<charconv>)

    struct from_chars_result : std::from_chars_result {
        [[nodiscard]] constexpr operator bool() const noexcept { return ec == std::errc{}; }
    };

    struct to_chars_result : std::to_chars_result {
        [[nodiscard]] constexpr operator bool() const noexcept { return ec == std::errc{}; }
    };

#else
    struct from_chars_result {
      const char* ptr;
      std::errc ec;

      [[nodiscard]] constexpr operator bool() const noexcept { return ec == std::errc{}; }
    };

    struct to_chars_result {
      char* ptr;
      std::errc ec;

      [[nodiscard]] constexpr operator bool() const noexcept { return ec == std::errc{}; }
    };
#endif

    // Max version string length = 5(<major>) + 1(.) + 5(<minor>) + 1(.) + 5(<patch>) + 1(-) + 5(<prerelease>) + 1(.) + 5(<prereleaseversion>) = 29.
    inline constexpr auto max_version_string_length = std::size_t{29};

    namespace detail {

        inline constexpr auto alpha = std::string_view{"alpha", 5};
        inline constexpr auto beta = std::string_view{"beta", 4};
        inline constexpr auto rc = std::string_view{"rc", 2};

        // Min version string length = 1(<major>) + 1(.) + 1(<minor>) + 1(.) + 1(<patch>) = 5.
        inline constexpr auto min_version_string_length = 5;

        constexpr std::uint8_t length(std::uint16_t x) noexcept {
            if (x < 10) {
                return 1;
            }
            if (x < 100) {
                return 2;
            }
            if (x < 1000) {
                return 3;
            }
            if (x < 10000) {
                return 4;
            }
            return 5;
        }

        constexpr std::uint8_t length(prerelease t) noexcept {
            if (t == prerelease::alpha) {
                return static_cast<std::uint8_t>(alpha.length());
            } else if (t == prerelease::beta) {
                return static_cast<std::uint8_t>(beta.length());
            } else if (t == prerelease::rc) {
                return static_cast<std::uint8_t>(rc.length());
            }

            return 0;
        }


        constexpr char *to_chars(char *str, std::uint16_t x, bool dot = true) noexcept {
            do {
                *(--str) = static_cast<char>('0' + (x % 10));
                x /= 10;
            } while (x != 0);

            if (dot) {
                *(--str) = '.';
            }

            return str;
        }

        constexpr char *to_chars(char *str, prerelease t) noexcept {
            const auto p = t == prerelease::alpha
                           ? alpha
                           : t == prerelease::beta
                             ? beta
                             : t == prerelease::rc
                               ? rc
                               : std::string_view{};

            if (p.size() > 0) {
                for (auto it = p.rbegin(); it != p.rend(); ++it) {
                    *(--str) = *it;
                }
                *(--str) = '-';
            }

            return str;
        }

        constexpr const char *from_chars(const char *first, const char *last, std::uint16_t &d) noexcept {
            if (first != last && ascii_is_digit(*first)) {
                std::int32_t t = 0;
                for (; first != last && ascii_is_digit(*first); ++first) {
                    t = t * 10 + ascii_to_digit(*first);
                }
                if (t <= (std::numeric_limits<std::uint16_t>::max)()) {
                    d = static_cast<std::uint16_t>(t);
                    return first;
                }
            }

            return nullptr;
        }

        constexpr const char *
        from_chars(const char *first, const char *last, std::optional<std::uint16_t> &d) noexcept {
            if (first != last && ascii_is_digit(*first)) {
                std::int32_t t = 0;
                for (; first != last && ascii_is_digit(*first); ++first) {
                    t = t * 10 + ascii_to_digit(*first);
                }
                if (t <= (std::numeric_limits<std::uint16_t>::max)()) {
                    d = static_cast<std::uint16_t>(t);
                    return first;
                }
            }

            return nullptr;
        }

        constexpr const char *from_chars(const char *first, const char *last, prerelease &p) noexcept {
            if (ascii_is_hyphen(*first)) {
                ++first;
            }

            if (str_equals_ignore_case(first, last, alpha)) {
                p = prerelease::alpha;
                return first + alpha.length();
            } else if (str_equals_ignore_case(first, last, beta)) {
                p = prerelease::beta;
                return first + beta.length();
            } else if (str_equals_ignore_case(first, last, rc)) {
                p = prerelease::rc;
                return first + rc.length();
            }

            return nullptr;
        }

        constexpr bool check_delimiter(const char *first, const char *last, char d) noexcept {
            return first != last && first != nullptr && *first == d;
        }

        template<typename T, typename = void>
        struct resize_uninitialized {
            static auto resize(T &str, std::size_t size) -> std::void_t<decltype(str.resize(size))> {
                str.resize(size);
            }
        };

        template<typename T>
        struct resize_uninitialized<T, std::void_t<decltype(std::declval<T>().__resize_default_init(42))>> {
            static void resize(T &str, std::size_t size) {
                str.__resize_default_init(size);
            }
        };

    } // namespace turbo::detail

    struct ModuleVersion {
        std::uint16_t major = 0;
        std::uint16_t minor = 1;
        std::uint16_t patch = 0;
        prerelease prerelease_type = prerelease::none;
        std::optional<std::uint16_t> prerelease_number = std::nullopt;

        constexpr ModuleVersion(std::uint16_t mj,
                                std::uint16_t mn,
                                std::uint16_t pt,
                                prerelease prt = prerelease::none,
                                std::optional<std::uint16_t> prn = std::nullopt) noexcept
                : major{mj},
                  minor{mn},
                  patch{pt},
                  prerelease_type{prt},
                  prerelease_number{prt == prerelease::none ? std::nullopt : prn} {
        }

        constexpr ModuleVersion(std::uint16_t mj,
                                std::uint16_t mn,
                                std::uint16_t pt,
                                prerelease prt,
                                std::uint16_t prn) noexcept
                : major{mj},
                  minor{mn},
                  patch{pt},
                  prerelease_type{prt},
                  prerelease_number{prt == prerelease::none ? std::nullopt : std::make_optional<std::uint16_t>(prn)} {
        }

        explicit constexpr ModuleVersion(std::string_view str) : ModuleVersion(0, 0, 0, prerelease::none,
                                                                               std::nullopt) {
            from_string(str);
        }

        constexpr ModuleVersion() = default;

        constexpr ModuleVersion(const ModuleVersion &) = default;

        constexpr ModuleVersion(ModuleVersion &&) = default;

        ~ModuleVersion() = default;

        ModuleVersion &operator=(const ModuleVersion &) = default;

        ModuleVersion &operator=(ModuleVersion &&) = default;

        [[nodiscard]] constexpr from_chars_result from_chars(const char *first, const char *last) noexcept {
            if (first == nullptr || last == nullptr || (last - first) < detail::min_version_string_length) {
                return {first, std::errc::invalid_argument};
            }

            auto next = first;
            if (next = detail::from_chars(next, last, major); detail::check_delimiter(next, last, '.')) {
                if (next = detail::from_chars(++next, last, minor); detail::check_delimiter(next, last, '.')) {
                    if (next = detail::from_chars(++next, last, patch); next == last) {
                        prerelease_type = prerelease::none;
                        prerelease_number = {};
                        return {next, std::errc{}};
                    } else if (detail::check_delimiter(next, last, '-')) {
                        if (next = detail::from_chars(next, last, prerelease_type); next == last) {
                            prerelease_number = {};
                            return {next, std::errc{}};
                        } else if (detail::check_delimiter(next, last, '.')) {
                            if (next = detail::from_chars(++next, last, prerelease_number); next == last) {
                                return {next, std::errc{}};
                            }
                        }
                    }
                }
            }

            return {first, std::errc::invalid_argument};
        }

        [[nodiscard]] constexpr to_chars_result to_chars(char *first, char *last) const noexcept {
            const auto length = string_length();
            if (first == nullptr || last == nullptr || (last - first) < length) {
                return {last, std::errc::value_too_large};
            }

            auto next = first + length;
            if (prerelease_type != prerelease::none) {
                if (prerelease_number.has_value()) {
                    next = detail::to_chars(next, prerelease_number.value());
                }
                next = detail::to_chars(next, prerelease_type);
            }
            next = detail::to_chars(next, patch);
            next = detail::to_chars(next, minor);
            next = detail::to_chars(next, major, false);

            return {first + length, std::errc{}};
        }

        [[nodiscard]] constexpr bool from_string_noexcept(std::string_view str) noexcept {
            return from_chars(str.data(), str.data() + str.length());
        }

        constexpr ModuleVersion &from_string(std::string_view str) {
            if (!from_string_noexcept(str)) {
                TURBO_MODULE_THROW("turbo::version::from_string invalid version.");
            }

            return *this;
        }

        [[nodiscard]] std::string to_string() const {
            auto str = std::string{};
            detail::resize_uninitialized<std::string>::resize(str, string_length());
            if (!to_chars(str.data(), str.data() + str.length())) {
                TURBO_MODULE_THROW("turbo::version::to_string invalid version.");
            }

            return str;
        }

        [[nodiscard]] constexpr std::uint8_t string_length() const noexcept {
            // (<major>) + 1(.) + (<minor>) + 1(.) + (<patch>)
            auto length = detail::length(major) + detail::length(minor) + detail::length(patch) + 2;
            if (prerelease_type != prerelease::none) {
                // + 1(-) + (<prerelease>)
                length += detail::length(prerelease_type) + 1;
                if (prerelease_number.has_value()) {
                    // + 1(.) + (<prereleaseversion>)
                    length += detail::length(prerelease_number.value()) + 1;
                }
            }

            return static_cast<std::uint8_t>(length);
        }

        [[nodiscard]] constexpr int compare(const ModuleVersion &other) const noexcept {
            if (major != other.major) {
                return major - other.major;
            }

            if (minor != other.minor) {
                return minor - other.minor;
            }

            if (patch != other.patch) {
                return patch - other.patch;
            }

            if (prerelease_type != other.prerelease_type) {
                return static_cast<std::uint8_t>(prerelease_type) - static_cast<std::uint8_t>(other.prerelease_type);
            }

            if (prerelease_number.has_value()) {
                if (other.prerelease_number.has_value()) {
                    return prerelease_number.value() - other.prerelease_number.value();
                }
                return 1;
            } else if (other.prerelease_number.has_value()) {
                return -1;
            }

            return 0;
        }
    };

    [[nodiscard]] constexpr bool operator==(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) == 0;
    }

    [[nodiscard]] constexpr bool operator!=(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) != 0;
    }

    [[nodiscard]] constexpr bool operator>(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) > 0;
    }

    [[nodiscard]] constexpr bool operator>=(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) >= 0;
    }

    [[nodiscard]] constexpr bool operator<(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) < 0;
    }

    [[nodiscard]] constexpr bool operator<=(const ModuleVersion &lhs, const ModuleVersion &rhs) noexcept {
        return lhs.compare(rhs) <= 0;
    }

    [[nodiscard]] constexpr ModuleVersion operator ""_version(const char *str, std::size_t length) {
        return ModuleVersion{std::string_view{str, length}};
    }

    [[nodiscard]] constexpr bool valid(std::string_view str) noexcept {
        return ModuleVersion{}.from_string_noexcept(str);
    }

    [[nodiscard]] constexpr from_chars_result
    from_chars(const char *first, const char *last, ModuleVersion &v) noexcept {
        return v.from_chars(first, last);
    }

    [[nodiscard]] constexpr to_chars_result to_chars(char *first, char *last, const ModuleVersion &v) noexcept {
        return v.to_chars(first, last);
    }

    [[nodiscard]] constexpr std::optional<ModuleVersion> from_string_noexcept(std::string_view str) noexcept {
        if (ModuleVersion v{}; v.from_string_noexcept(str)) {
            return v;
        }

        return std::nullopt;
    }

    [[nodiscard]] constexpr ModuleVersion from_string(std::string_view str) {
        return ModuleVersion{str};
    }

    [[nodiscard]] inline std::string to_string(const ModuleVersion &v) {
        return v.to_string();
    }

    template<typename Char, typename Traits>
    inline std::basic_ostream<Char, Traits> &operator<<(std::basic_ostream<Char, Traits> &os, const ModuleVersion &v) {
        for (const auto c: v.to_string()) {
            os.put(c);
        }

        return os;
    }

    inline namespace comparators {

        enum struct comparators_option : std::uint8_t {
            exclude_prerelease,
            include_prerelease
        };

        [[nodiscard]] constexpr int compare(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                            comparators_option option = comparators_option::include_prerelease) noexcept {
            if (option == comparators_option::exclude_prerelease) {
                return ModuleVersion{lhs.major, lhs.minor, lhs.patch}.compare(
                        ModuleVersion{rhs.major, rhs.minor, rhs.patch});
            }
            return lhs.compare(rhs);
        }

        [[nodiscard]] constexpr bool equal_to(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                              comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) == 0;
        }

        [[nodiscard]] constexpr bool not_equal_to(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                                  comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) != 0;
        }

        [[nodiscard]] constexpr bool greater(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                             comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) > 0;
        }

        [[nodiscard]] constexpr bool greater_equal(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                                   comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) >= 0;
        }

        [[nodiscard]] constexpr bool less(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                          comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) < 0;
        }

        [[nodiscard]] constexpr bool less_equal(const ModuleVersion &lhs, const ModuleVersion &rhs,
                                                comparators_option option = comparators_option::include_prerelease) noexcept {
            return compare(lhs, rhs, option) <= 0;
        }

    } // namespace turbo::comparators

    namespace range {

        namespace detail {

            using namespace turbo::detail;

            class range {
            public:
                constexpr explicit range(std::string_view str) noexcept: parser{str} {}

                constexpr bool satisfies(const ModuleVersion &ver, bool include_prerelease) {
                    const bool has_prerelease = ver.prerelease_type != prerelease::none;

                    do {
                        if (is_logical_or_token()) {
                            parser.advance_token(range_token_type::logical_or);
                        }

                        bool contains = true;
                        bool allow_compare = include_prerelease;

                        while (is_operator_token() || is_number_token()) {
                            const auto range = parser.parse_range();
                            const bool equal_without_tags = equal_to(range.ver, ver,
                                                                     comparators_option::exclude_prerelease);

                            if (has_prerelease && equal_without_tags) {
                                allow_compare = true;
                            }

                            if (!range.satisfies(ver)) {
                                contains = false;
                                break;
                            }
                        }

                        if (has_prerelease) {
                            if (allow_compare && contains) {
                                return true;
                            }
                        } else if (contains) {
                            return true;
                        }

                    } while (is_logical_or_token());

                    return false;
                }

            private:
                enum struct range_operator : std::uint8_t {
                    less,
                    less_or_equal,
                    greater,
                    greater_or_equal,
                    equal
                };

                struct range_comparator {
                    range_operator op;
                    ModuleVersion ver;

                    constexpr bool satisfies(const ModuleVersion &version) const {
                        switch (op) {
                            case range_operator::equal:
                                return version == ver;
                            case range_operator::greater:
                                return version > ver;
                            case range_operator::greater_or_equal:
                                return version >= ver;
                            case range_operator::less:
                                return version < ver;
                            case range_operator::less_or_equal:
                                return version <= ver;
                            default:
                                TURBO_MODULE_THROW("turbo::range unexpected operator.");
                        }
                    }
                };

                enum struct range_token_type : std::uint8_t {
                    none,
                    number,
                    range_operator,
                    dot,
                    logical_or,
                    hyphen,
                    prerelease,
                    end_of_line
                };

                struct range_token {
                    range_token_type type = range_token_type::none;
                    std::uint16_t number = 0;
                    range_operator op = range_operator::equal;
                    prerelease prerelease_type = prerelease::none;
                };

                struct range_lexer {
                    std::string_view text;
                    std::size_t pos;

                    constexpr explicit range_lexer(std::string_view text) noexcept: text{text}, pos{0} {}

                    constexpr range_token get_next_token() noexcept {
                        while (!end_of_line()) {

                            if (ascii_is_space(text[pos])) {
                                advance(1);
                                continue;
                            }

                            if (ascii_is_logical_or(text[pos])) {
                                advance(2);
                                return {range_token_type::logical_or};
                            }

                            if (ascii_is_operator(text[pos])) {
                                const auto op = get_operator();
                                return {range_token_type::range_operator, 0, op};
                            }

                            if (ascii_is_digit(text[pos])) {
                                const auto number = get_number();
                                return {range_token_type::number, number};
                            }

                            if (ascii_is_dot(text[pos])) {
                                advance(1);
                                return {range_token_type::dot};
                            }

                            if (ascii_is_hyphen(text[pos])) {
                                advance(1);
                                return {range_token_type::hyphen};
                            }

                            if (ascii_is_letter(text[pos])) {
                                const auto prerelease = get_prerelease();
                                return {range_token_type::prerelease, 0, range_operator::equal, prerelease};
                            }
                        }

                        return {range_token_type::end_of_line};
                    }

                    constexpr bool end_of_line() const noexcept { return pos >= text.length(); }

                    constexpr void advance(std::size_t i) noexcept {
                        pos += i;
                    }

                    constexpr range_operator get_operator() noexcept {
                        if (text[pos] == '<') {
                            advance(1);
                            if (text[pos] == '=') {
                                advance(1);
                                return range_operator::less_or_equal;
                            }
                            return range_operator::less;
                        } else if (text[pos] == '>') {
                            advance(1);
                            if (text[pos] == '=') {
                                advance(1);
                                return range_operator::greater_or_equal;
                            }
                            return range_operator::greater;
                        } else if (text[pos] == '=') {
                            advance(1);
                            return range_operator::equal;
                        }

                        return range_operator::equal;
                    }

                    constexpr std::uint16_t get_number() noexcept {
                        const auto first = text.data() + pos;
                        const auto last = text.data() + text.length();
                        if (std::uint16_t n{}; from_chars(first, last, n) != nullptr) {
                            advance(length(n));
                            return n;
                        }

                        return 0;
                    }

                    constexpr prerelease get_prerelease() noexcept {
                        const auto first = text.data() + pos;
                        const auto last = text.data() + text.length();
                        if (first > last) {
                            advance(1);
                            return prerelease::none;
                        }

                        if (prerelease p{}; from_chars(first, last, p) != nullptr) {
                            advance(length(p));
                            return p;
                        }

                        advance(1);

                        return prerelease::none;
                    }
                };

                struct range_parser {
                    range_lexer lexer;
                    range_token current_token;

                    constexpr explicit range_parser(std::string_view str) : lexer{str},
                                                                            current_token{range_token_type::none} {
                        advance_token(range_token_type::none);
                    }

                    constexpr void advance_token(range_token_type token_type) {
                        if (current_token.type != token_type) {
                            TURBO_MODULE_THROW("turbo::range unexpected token.");
                        }
                        current_token = lexer.get_next_token();
                    }

                    constexpr range_comparator parse_range() {
                        if (current_token.type == range_token_type::number) {
                            const auto version = parse_version();
                            return {range_operator::equal, version};
                        } else if (current_token.type == range_token_type::range_operator) {
                            const auto range_operator = current_token.op;
                            advance_token(range_token_type::range_operator);
                            const auto version = parse_version();
                            return {range_operator, version};
                        }

                        return {range_operator::equal, ModuleVersion{}};
                    }

                    constexpr ModuleVersion parse_version() {
                        const auto major = parse_number();

                        advance_token(range_token_type::dot);
                        const auto minor = parse_number();

                        advance_token(range_token_type::dot);
                        const auto patch = parse_number();

                        prerelease prerelease = prerelease::none;
                        std::optional<std::uint16_t> prerelease_number = std::nullopt;

                        if (current_token.type == range_token_type::hyphen) {
                            advance_token(range_token_type::hyphen);
                            prerelease = parse_prerelease();
                            if (current_token.type == range_token_type::dot) {
                                advance_token(range_token_type::dot);
                                prerelease_number = parse_number();
                            }
                        }

                        return {major, minor, patch, prerelease, prerelease_number};
                    }

                    constexpr std::uint16_t parse_number() {
                        const auto token = current_token;
                        advance_token(range_token_type::number);

                        return token.number;
                    }

                    constexpr prerelease parse_prerelease() {
                        const auto token = current_token;
                        advance_token(range_token_type::prerelease);

                        return token.prerelease_type;
                    }
                };

                [[nodiscard]] constexpr bool is_logical_or_token() const noexcept {
                    return parser.current_token.type == range_token_type::logical_or;
                }

                [[nodiscard]] constexpr bool is_operator_token() const noexcept {
                    return parser.current_token.type == range_token_type::range_operator;
                }

                [[nodiscard]] constexpr bool is_number_token() const noexcept {
                    return parser.current_token.type == range_token_type::number;
                }

                range_parser parser;
            };

        } // namespace turbo::range::detail

        enum struct satisfies_option : std::uint8_t {
            exclude_prerelease,
            include_prerelease
        };

        constexpr bool satisfies(const ModuleVersion &ver, std::string_view str,
                                 satisfies_option option = satisfies_option::exclude_prerelease) {
            switch (option) {
                case satisfies_option::exclude_prerelease:
                    return detail::range{str}.satisfies(ver, false);
                case satisfies_option::include_prerelease:
                    return detail::range{str}.satisfies(ver, true);
                default:
                    TURBO_MODULE_THROW("turbo::range unexpected satisfies_option.");
            }
        }

    } // namespace turbo::range

} // namespace turbo


template<>
struct turbo::formatter<turbo::ModuleVersion> {
    // show major version
    bool show_major = false;
    // show minor version
    bool show_minor = false;
    // show patch version
    bool show_patch = false;

    // Parses format specifications of the form ['M' | 'm' | 'p'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end || *it == '}') {
            show_major = show_minor = show_patch = true;
            return it;
        }
        do {
            switch (*it) {
                case 'M': show_major = true; break;
                case 'm': show_minor = true; break;
                case 'p': show_patch = true; break;
                default: throw format_error("invalid format");
            }
            ++it;
        } while (it != end && *it != '}');
        return it;
    }

    template<typename FormatContext>
    auto format(const turbo::ModuleVersion& ver, FormatContext& ctx)
    -> decltype(ctx.out()) {
        if (ver.major == uint16_t(-1)) return format_to(ctx.out(), "N/A");
        if (ver.minor == uint16_t(-1)) show_minor = false;
        if (ver.patch == uint16_t(-1)) show_patch = false;
        if (show_major && !show_minor && !show_patch) {
            return format_to(ctx.out(), "{}", ver.major);
        }
        if (show_major && show_minor && !show_patch) {
            return format_to(ctx.out(), "{}.{}", ver.major, ver.minor);
        }
        if (show_major && show_minor && show_patch) {
            return format_to(ctx.out(), "{}.{}.{}", ver.major, ver.minor,
                             ver.patch);
        }
        return ctx.out();
    }
};

#if defined(__clang__)
#  pragma clang diagnostic pop
#endif

#endif // TURBO_MODULE_MODULE_VERSION_H_
