// Copyright 2023 The titan-search Authors.
// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
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

#include <turbo/log/tweakme.h>
#include "turbo/log/details/null_mutex.h"
#include "turbo/files/filesystem.h"
#include "turbo/times/clock.h"

#include <atomic>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <exception>
#include <string>
#include <type_traits>
#include <functional>
#include <cstdio>
#include <string_view>
#include <turbo/format/format.h>

// disable thread local on msvc 2013
#ifndef TLOG_NO_TLS
#    if (defined(_MSC_VER) && (_MSC_VER < 1900)) || defined(__cplusplus_winrt)
#        define TLOG_NO_TLS 1
#    endif
#endif

#ifndef TLOG_FUNCTION
#    define TLOG_FUNCTION static_cast<const char *>(__FUNCTION__)
#endif

#ifdef TLOG_NO_EXCEPTIONS
#    define TLOG_TRY
#    define TLOG_THROW(ex)                                                                                                               \
        do                                                                                                                                 \
        {                                                                                                                                  \
            printf("tlog fatal error: %s\n", ex.what());                                                                                 \
            std::abort();                                                                                                                  \
        } while (0)
#    define TLOG_CATCH_STD
#else
#    define TLOG_TRY try
#    define TLOG_THROW(ex) throw(ex)
#    define TLOG_CATCH_STD                                                                                                               \
        catch (const std::exception &) {}
#endif

namespace turbo::tlog {

    class formatter;

    namespace sinks {
        class sink;
    }

#if defined(_WIN32) && defined(TLOG_WCHAR_FILENAMES)
    using filename_t = std::wstring;
    // allow macro expansion to occur in TLOG_FILENAME_T
#    define TLOG_FILENAME_T_INNER(s) L##s
#    define TLOG_FILENAME_T(s) TLOG_FILENAME_T_INNER(s)
#else
    using filename_t = std::string;
#    define TLOG_FILENAME_T(s) s
#endif

    using sink_ptr = std::shared_ptr<sinks::sink>;
    using sinks_init_list = std::initializer_list<sink_ptr>;
    using err_handler = std::function<void(const std::string &err_msg)>;

    using memory_buf_t = turbo::basic_memory_buffer<char, 250>;

    template<typename... Args>
    using format_string_t = turbo::format_string<Args...>;

    template<class T>
    using remove_cvref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

// clang doesn't like SFINAE disabled constructor in std::is_convertible<> so have to repeat the condition from basic_format_string here,
// in addition, turbo::basic_runtime<Char> is only convertible to basic_format_string<Char> but not basic_string_view<Char>
    template<class T, class Char = char>
    struct is_convertible_to_basic_format_string
            : std::integral_constant<bool,
                    std::is_convertible<T, std::basic_string_view<Char>>::value> {
    };

#    if defined(TLOG_WCHAR_FILENAMES) || defined(TLOG_WCHAR_TO_UTF8_SUPPORT)
    using wstring_view_t = turbo::basic_string_view<wchar_t>;
    using wmemory_buf_t = turbo::basic_memory_buffer<wchar_t, 250>;

    template<typename... Args>
    using wformat_string_t = turbo::wformat_string<Args...>;
#    endif
#    define TLOG_BUF_TO_STRING(x) turbo::to_string(x)

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT
#    ifndef _WIN32
#        error TLOG_WCHAR_TO_UTF8_SUPPORT only supported on windows
#    endif // _WIN32
#endif     // TLOG_WCHAR_TO_UTF8_SUPPORT

    template<class T>
    struct is_convertible_to_any_format_string : std::integral_constant<bool,
            is_convertible_to_basic_format_string<T, char>::value ||
            is_convertible_to_basic_format_string<T, wchar_t>::value> {
    };

#if defined(TLOG_NO_ATOMIC_LEVELS)
    using level_t = details::null_atomic_int;
#else
    using level_t = std::atomic<int>;
#endif

#define TLOG_LEVEL_TRACE 0
#define TLOG_LEVEL_DEBUG 1
#define TLOG_LEVEL_INFO 2
#define TLOG_LEVEL_WARN 3
#define TLOG_LEVEL_ERROR 4
#define TLOG_LEVEL_CRITICAL 5
#define TLOG_LEVEL_OFF 6

#if !defined(TLOG_ACTIVE_LEVEL)
#    define TLOG_ACTIVE_LEVEL TLOG_LEVEL_TRACE
#endif

// Log level enum
    namespace level {
        enum level_enum : int {
            trace = TLOG_LEVEL_TRACE,
            debug = TLOG_LEVEL_DEBUG,
            info = TLOG_LEVEL_INFO,
            warn = TLOG_LEVEL_WARN,
            err = TLOG_LEVEL_ERROR,
            critical = TLOG_LEVEL_CRITICAL,
            off = TLOG_LEVEL_OFF,
            n_levels
        };

#define TLOG_LEVEL_NAME_TRACE std::string_view("trace", 5)
#define TLOG_LEVEL_NAME_DEBUG std::string_view("debug", 5)
#define TLOG_LEVEL_NAME_INFO std::string_view("info", 4)
#define TLOG_LEVEL_NAME_WARNING std::string_view("warning", 7)
#define TLOG_LEVEL_NAME_ERROR std::string_view("error", 5)
#define TLOG_LEVEL_NAME_CRITICAL std::string_view("critical", 8)
#define TLOG_LEVEL_NAME_OFF std::string_view("off", 3)

#if !defined(TLOG_LEVEL_NAMES)
#    define TLOG_LEVEL_NAMES                                                                                                             \
        {                                                                                                                                  \
            TLOG_LEVEL_NAME_TRACE, TLOG_LEVEL_NAME_DEBUG, TLOG_LEVEL_NAME_INFO, TLOG_LEVEL_NAME_WARNING, TLOG_LEVEL_NAME_ERROR,  \
                TLOG_LEVEL_NAME_CRITICAL, TLOG_LEVEL_NAME_OFF                                                                          \
        }
#endif

#if !defined(TLOG_SHORT_LEVEL_NAMES)

#    define TLOG_SHORT_LEVEL_NAMES                                                                                                       \
        {                                                                                                                                  \
            "T", "D", "I", "W", "E", "C", "O"                                                                                              \
        }
#endif

        TURBO_DLL const std::string_view &to_string_view(turbo::tlog::level::level_enum l) noexcept;

        TURBO_DLL const char *to_short_c_str(turbo::tlog::level::level_enum l) noexcept;

        TURBO_DLL turbo::tlog::level::level_enum from_str(const std::string &name) noexcept;

    } // namespace level

    //
    // Color mode used by sinks with color support.
    //
    enum class color_mode {
        always,
        automatic,
        never
    };

    //
    // Pattern time - specific time getting to use for pattern_formatter.
    // local time by default
    //
    enum class pattern_time_type {
        local, // log localtime
        utc    // log utc
    };

    //
    // Log exception
    //
    class TURBO_DLL tlog_ex : public std::exception {
    public:
        explicit tlog_ex(std::string msg);

        tlog_ex(const std::string &msg, int last_errno);

        const char *what() const noexcept override;

    private:
        std::string msg_;
    };

    [[noreturn]] TURBO_DLL void throw_tlog_ex(const std::string &msg, int last_errno);

    [[noreturn]] TURBO_DLL void throw_tlog_ex(std::string msg);

    struct source_loc {
        constexpr source_loc() = default;

        constexpr source_loc(const char *filename_in, int line_in, const char *funcname_in)
                : filename{filename_in}, line{line_in}, funcname{funcname_in} {}

        constexpr bool empty() const noexcept {
            return line == 0;
        }

        const char *filename{nullptr};
        int line{0};
        const char *funcname{nullptr};
    };

    namespace details {

        using std::enable_if_t;
        using std::make_unique;

        // to avoid useless casts (see https://github.com/nlohmann/json/issues/2893#issuecomment-889152324)
        template<typename T, typename U, enable_if_t<!std::is_same<T, U>::value, int> = 0>
        constexpr T conditional_static_cast(U value) {
            return static_cast<T>(value);
        }

        template<typename T, typename U, enable_if_t<std::is_same<T, U>::value, int> = 0>
        constexpr T conditional_static_cast(U value) {
            return value;
        }

    } // namespace details

    constexpr turbo::OpenOption get_append_option() {
        turbo::OpenOption option = turbo::kDefaultAppendWriteOption;
        return option.create(true).append(true).tries(5).interval(10).create_dir(true);
    }

    static constexpr turbo::OpenOption  kLogAppendOpenOption = get_append_option();

    static constexpr turbo::OpenOption  kLogTruncateOpenOption = []() {
        turbo::OpenOption option = turbo::kDefaultTruncateWriteOption;
        return option.create(true).truncate(true).tries(5).interval(10).create_dir(true);
    }();

} // namespace turbo::tlog

