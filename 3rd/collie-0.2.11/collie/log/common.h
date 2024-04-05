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

#include <collie/log/details/null_mutex.h>
#include <collie/log/tweakme.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>


#include <collie/strings/fmt/format.h>
#include <collie/strings/fmt/xchar.h>

#ifndef CLOG_FUNCTION
#define CLOG_FUNCTION static_cast<const char *>(__FUNCTION__)
#endif

#ifdef CLOG_NO_EXCEPTIONS
#define CLOG_TRY
#define CLOG_THROW(ex)                               \
        do {                                               \
            printf("clog fatal error: %s\n", ex.what()); \
            std::abort();                                  \
        } while (0)
#define CLOG_CATCH_STD
#else
#define CLOG_TRY try
#define CLOG_THROW(ex) throw(ex)
#define CLOG_CATCH_STD             \
        catch (const std::exception &) { \
        }
#endif

namespace clog {

    class formatter;

    namespace sinks {
        class sink;
    }

#if defined(_WIN32) && defined(CLOG_WCHAR_FILENAMES)
    using filename_t = std::wstring;
        // allow macro expansion to occur in CLOG_FILENAME_T
#define CLOG_FILENAME_T_INNER(s) L##s
#define CLOG_FILENAME_T(s) CLOG_FILENAME_T_INNER(s)
#else
    using filename_t = std::string;
#define CLOG_FILENAME_T(s) s
#endif

    using log_clock = std::chrono::system_clock;
    using sink_ptr = std::shared_ptr<sinks::sink>;
    using sinks_init_list = std::initializer_list<sink_ptr>;
    using err_handler = std::function<void(const std::string &err_msg)>;

    namespace fmt_lib = fmt;

    using string_view_t = fmt::basic_string_view<char>;
    using memory_buf_t = fmt::basic_memory_buffer<char, 250>;

    template<typename... Args>
    using format_string_t = fmt::format_string<Args...>;

    template<class T>
    using remove_cvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    template<typename Char>
    using fmt_runtime_string = fmt::runtime_format_string<Char>;

    // clang doesn't like SFINAE disabled constructor in std::is_convertible<> so have to repeat the
    // condition from basic_format_string here, in addition, fmt::basic_runtime<Char> is only
    // convertible to basic_format_string<Char> but not basic_string_view<Char>
    template<class T, class Char = char>
    struct is_convertible_to_basic_format_string
            : std::integral_constant<bool,
                    std::is_convertible<T, fmt::basic_string_view<Char>>::value ||
                    std::is_same<remove_cvref_t<T>, fmt_runtime_string<Char>>::value> {
    };

#if defined(CLOG_WCHAR_FILENAMES) || defined(CLOG_WCHAR_TO_UTF8_SUPPORT)
    using wstring_view_t = fmt::basic_string_view<wchar_t>;
    using wmemory_buf_t = fmt::basic_memory_buffer<wchar_t, 250>;

    template <typename... Args>
    using wformat_string_t = fmt::wformat_string<Args...>;
#endif

#ifdef CLOG_WCHAR_TO_UTF8_SUPPORT
#ifndef _WIN32
#error CLOG_WCHAR_TO_UTF8_SUPPORT only supported on windows
#endif  // _WIN32
#endif      // CLOG_WCHAR_TO_UTF8_SUPPORT

    template<class T>
    struct is_convertible_to_any_format_string
            : std::integral_constant<bool,
                    is_convertible_to_basic_format_string<T, char>::value ||
                    is_convertible_to_basic_format_string<T, wchar_t>::value> {
    };

#if defined(CLOG_NO_ATOMIC_LEVELS)
    using level_t = details::null_atomic_int;
#else
    using level_t = std::atomic<int>;
#endif

#define CLOG_LEVEL_TRACE 0
#define CLOG_LEVEL_DEBUG 1
#define CLOG_LEVEL_INFO 2
#define CLOG_LEVEL_WARN 3
#define CLOG_LEVEL_ERROR 4
#define CLOG_LEVEL_FATAL 5
#define CLOG_LEVEL_OFF 6

#if !defined(CLOG_ACTIVE_LEVEL)
#define CLOG_ACTIVE_LEVEL CLOG_LEVEL_INFO
#endif

    // Log level enum
    namespace level {
        enum level_enum : int {
            trace = CLOG_LEVEL_TRACE,
            debug = CLOG_LEVEL_DEBUG,
            info = CLOG_LEVEL_INFO,
            warn = CLOG_LEVEL_WARN,
            error = CLOG_LEVEL_ERROR,
            fatal = CLOG_LEVEL_FATAL,
            off = CLOG_LEVEL_OFF,
            n_levels
        };

#define CLOG_LEVEL_NAME_TRACE clog::string_view_t("trace", 5)
#define CLOG_LEVEL_NAME_DEBUG clog::string_view_t("debug", 5)
#define CLOG_LEVEL_NAME_INFO clog::string_view_t("info", 4)
#define CLOG_LEVEL_NAME_WARN clog::string_view_t("warn", 4)
#define CLOG_LEVEL_NAME_ERROR clog::string_view_t("error", 5)
#define CLOG_LEVEL_NAME_FATAL clog::string_view_t("fatal", 5)
#define CLOG_LEVEL_NAME_OFF clog::string_view_t("off", 3)

#if !defined(CLOG_LEVEL_NAMES)
#define CLOG_LEVEL_NAMES                                                                  \
        {                                                                                       \
            CLOG_LEVEL_NAME_TRACE, CLOG_LEVEL_NAME_DEBUG, CLOG_LEVEL_NAME_INFO,           \
                CLOG_LEVEL_NAME_WARN, CLOG_LEVEL_NAME_ERROR, CLOG_LEVEL_NAME_FATAL, \
                CLOG_LEVEL_NAME_OFF                                                           \
        }
#endif

#if !defined(CLOG_SHORT_LEVEL_NAMES)

#define CLOG_SHORT_LEVEL_NAMES \
        { "T", "D", "I", "W", "E", "C", "O" }
#endif

        const string_view_t &to_string_view(clog::level::level_enum l) noexcept;

        const char *to_short_c_str(clog::level::level_enum l) noexcept;

        clog::level::level_enum from_str(const std::string &name) noexcept;

    }  // namespace level

    //
    // Color mode used by sinks with color support.
    //
    enum class color_mode {
        always, automatic, never
    };

    //
    // Pattern time - specific time getting to use for pattern_formatter.
    // local time by default
    //
    enum class pattern_time_type {
        local,  // log localtime
        utc     // log utc
    };

    //
    // Log exception
    //
    class CLogEx : public std::exception {
    public:
        explicit CLogEx(std::string msg);

        CLogEx(const std::string &msg, int last_errno);

        const char *what() const noexcept override;

    private:
        std::string msg_;
    };

    [[noreturn]]  void throw_clog_ex(const std::string &msg, int last_errno);

    [[noreturn]]  void throw_clog_ex(std::string msg);

    struct source_loc {
        constexpr source_loc() = default;

        constexpr source_loc(const char *filename_in, int line_in, const char *funcname_in)
                : filename{filename_in},
                  line{line_in},
                  funcname{funcname_in} {}

        constexpr bool empty() const noexcept { return line == 0; }

        const char *filename{nullptr};
        int line{0};
        const char *funcname{nullptr};
    };

    struct file_event_handlers {
        file_event_handlers()
                : before_open(nullptr),
                  after_open(nullptr),
                  before_close(nullptr),
                  after_close(nullptr) {}

        std::function<void(const filename_t &filename)> before_open;
        std::function<void(const filename_t &filename, std::FILE *file_stream)> after_open;
        std::function<void(const filename_t &filename, std::FILE *file_stream)> before_close;
        std::function<void(const filename_t &filename)> after_close;
    };

    namespace details {

        // to_string_view

        constexpr clog::string_view_t to_string_view(const memory_buf_t &buf) noexcept {
            return clog::string_view_t{buf.data(), buf.size()};
        }

        constexpr clog::string_view_t to_string_view(clog::string_view_t str)
        noexcept {
            return str;
        }

#if defined(CLOG_WCHAR_FILENAMES) || defined(CLOG_WCHAR_TO_UTF8_SUPPORT)
        constexpr clog::wstring_view_t to_string_view(const wmemory_buf_t &buf)
            noexcept {
            return clog::wstring_view_t{buf.data(), buf.size()};
        }

        constexpr clog::wstring_view_t to_string_view(clog::wstring_view_t str)
            noexcept {
            return str;
        }
#endif

        template<typename T, typename... Args>
        inline fmt::basic_string_view<T> to_string_view(fmt::basic_format_string<T, Args...> fmt) {
            return fmt;
        }

    // make_unique support for pre c++14
#if __cplusplus >= 201402L  // C++14 and beyond
        using std::enable_if_t;
        using std::make_unique;
#else
        template <bool B, class T = void>
        using enable_if_t = typename std::enable_if<B, T>::type;

        template <typename T, typename... Args>
        std::unique_ptr<T> make_unique(Args &&...args) {
            static_assert(!std::is_array<T>::value, "arrays not supported");
            return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        }
#endif

        // to avoid useless casts (see https://github.com/nlohmann/json/issues/2893#issuecomment-889152324)
        template<typename T, typename U, enable_if_t<!std::is_same<T, U>::value, int> = 0>
        constexpr T conditional_static_cast(U value) {
            return static_cast<T>(value);
        }

        template<typename T, typename U, enable_if_t<std::is_same<T, U>::value, int> = 0>
        constexpr T conditional_static_cast(U value) {
            return value;
        }

    }  // namespace details
}  // namespace clog

#include <collie/log/common-inl.h>
