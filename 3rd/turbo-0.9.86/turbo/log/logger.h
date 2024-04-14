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

// Thread safe logger (except for set_error_handler())
// Has name, log level, vector of std::shared sink pointers and formatter
// Upon each log write the logger:
// 1. Checks if its log level is enough to log the message and if yes:
// 2. Call the underlying sinks to do the job.
// 3. Each sink use its own private copy of a formatter to format the message
// and send to its destination.
//
// The use of private formatter per sink provides the opportunity to cache some
// formatted data, and support for different format per sink.

#include "turbo/log/common.h"
#include "turbo/log/details/log_msg.h"
#include <turbo/log/details/backtracer.h>

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT
#    ifndef _WIN32
#        error TLOG_WCHAR_TO_UTF8_SUPPORT only supported on windows
#    endif
#    include <turbo/log/details/os.h>
#endif

#include <vector>

#ifndef TLOG_NO_EXCEPTIONS
#    define TLOG_LOGGER_CATCH(location)                                                                                                  \
        catch (const std::exception &ex)                                                                                                   \
        {                                                                                                                                  \
            if (location.filename)                                                                                                         \
            {                                                                                                                              \
                err_handler_(turbo::format(FMT_STRING("{} [{}({})]"), ex.what(), location.filename, location.line));              \
            }                                                                                                                              \
            else                                                                                                                           \
            {                                                                                                                              \
                err_handler_(ex.what());                                                                                                   \
            }                                                                                                                              \
        }                                                                                                                                  \
        catch (...)                                                                                                                        \
        {                                                                                                                                  \
            err_handler_("Rethrowing unknown exception in logger");                                                                        \
            throw;                                                                                                                         \
        }
#else
#    define TLOG_LOGGER_CATCH(location)
#endif

namespace turbo::tlog {

    class TURBO_DLL logger {
    public:
        // Empty logger
        explicit logger(std::string name)
                : name_(std::move(name)), sinks_() {}

        // Logger with range on sinks
        template<typename It>
        logger(std::string name, It begin, It end)
                : name_(std::move(name)), sinks_(begin, end) {}

        // Logger with single sink
        logger(std::string name, sink_ptr single_sink)
                : logger(std::move(name), {std::move(single_sink)}) {}

        // Logger with sinks init list
        logger(std::string name, sinks_init_list sinks)
                : logger(std::move(name), sinks.begin(), sinks.end()) {}

        virtual ~logger() = default;

        logger(const logger &other);

        logger(logger &&other) noexcept;

        logger &operator=(logger other) noexcept;

        void swap(turbo::tlog::logger &other) noexcept;

        template<typename... Args>
        void log(source_loc loc, level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args) {
            log_(loc, lvl, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void log(level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args) {
            log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
        }

        template<typename T>
        void log(level::level_enum lvl, const T &msg) {
            log(source_loc{}, lvl, msg);
        }

        // T cannot be statically converted to format string (including string_view/wstring_view)
        template<class T, typename std::enable_if<!is_convertible_to_any_format_string<const T &>::value, int>::type = 0>
        void log(source_loc loc, level::level_enum lvl, const T &msg) {
            log(loc, lvl, "{}", msg);
        }

        void log(turbo::Time log_time, source_loc loc, level::level_enum lvl, std::string_view msg) {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled) {
                return;
            }

            details::log_msg log_msg(log_time, loc, name_, lvl, msg);
            log_it_(log_msg, log_enabled, traceback_enabled);
        }

        void log(source_loc loc, level::level_enum lvl, std::string_view msg) {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled) {
                return;
            }

            details::log_msg log_msg(loc, name_, lvl, msg);
            log_it_(log_msg, log_enabled, traceback_enabled);
        }

        void log(level::level_enum lvl, std::string_view msg) {
            log(source_loc{}, lvl, msg);
        }

        template<typename... Args>
        void trace(format_string_t<Args...> fmt, Args &&... args) {
            log(level::trace, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void debug(format_string_t<Args...> fmt, Args &&... args) {
            log(level::debug, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void info(format_string_t<Args...> fmt, Args &&... args) {
            log(level::info, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void warn(format_string_t<Args...> fmt, Args &&... args) {
            log(level::warn, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void error(format_string_t<Args...> fmt, Args &&... args) {
            log(level::err, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void critical(format_string_t<Args...> fmt, Args &&... args) {
            log(level::critical, fmt, std::forward<Args>(args)...);
        }

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT
        template<typename... Args>
        void log(source_loc loc, level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
        {
            log_(loc, lvl, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void log(level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
        }

        void log(log_clock::time_point log_time, source_loc loc, level::level_enum lvl, wstring_view_t msg)
        {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled)
            {
                return;
            }

            memory_buf_t buf;
            details::os::wstr_to_utf8buf(wstring_view_t(msg.data(), msg.size()), buf);
            details::log_msg log_msg(log_time, loc, name_, lvl, std::string_view(buf.data(), buf.size()));
            log_it_(log_msg, log_enabled, traceback_enabled);
        }

        void log(source_loc loc, level::level_enum lvl, wstring_view_t msg)
        {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled)
            {
                return;
            }

            memory_buf_t buf;
            details::os::wstr_to_utf8buf(wstring_view_t(msg.data(), msg.size()), buf);
            details::log_msg log_msg(loc, name_, lvl, std::string_view(buf.data(), buf.size()));
            log_it_(log_msg, log_enabled, traceback_enabled);
        }

        void log(level::level_enum lvl, wstring_view_t msg)
        {
            log(source_loc{}, lvl, msg);
        }

        template<typename... Args>
        void trace(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::trace, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void debug(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::debug, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void info(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::info, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void warn(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::warn, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void error(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::err, fmt, std::forward<Args>(args)...);
        }

        template<typename... Args>
        void critical(wformat_string_t<Args...> fmt, Args &&... args)
        {
            log(level::critical, fmt, std::forward<Args>(args)...);
        }
#endif

        template<typename T>
        void trace(const T &msg) {
            log(level::trace, msg);
        }

        template<typename T>
        void debug(const T &msg) {
            log(level::debug, msg);
        }

        template<typename T>
        void info(const T &msg) {
            log(level::info, msg);
        }

        template<typename T>
        void warn(const T &msg) {
            log(level::warn, msg);
        }

        template<typename T>
        void error(const T &msg) {
            log(level::err, msg);
        }

        template<typename T>
        void critical(const T &msg) {
            log(level::critical, msg);
        }

        // return true logging is enabled for the given level.
        bool should_log(level::level_enum msg_level) const {
            return msg_level >= level_.load(std::memory_order_relaxed);
        }

        // return true if backtrace logging is enabled.
        bool should_backtrace() const {
            return tracer_.enabled();
        }

        void set_level(level::level_enum log_level);

        level::level_enum level() const;

        const std::string &name() const;

        // set formatting for the sinks in this logger.
        // each sink will get a separate instance of the formatter object.
        void set_formatter(std::unique_ptr<formatter> f);

        // set formatting for the sinks in this logger.
        // equivalent to
        //     set_formatter(make_unique<pattern_formatter>(pattern, time_type))
        // Note: each sink will get a new instance of a formatter object, replacing the old one.
        void set_pattern(std::string pattern, pattern_time_type time_type = pattern_time_type::local);

        // backtrace support.
        // efficiently store all debug/trace messages in a circular buffer until needed for debugging.
        void enable_backtrace(size_t n_messages);

        void disable_backtrace();

        void dump_backtrace();

        // flush functions
        void flush();

        void flush_on(level::level_enum log_level);

        level::level_enum flush_level() const;

        // sinks
        const std::vector<sink_ptr> &sinks() const;

        std::vector<sink_ptr> &sinks();

        // error handler
        void set_error_handler(err_handler);

        // create new logger with same sinks and configuration.
        virtual std::shared_ptr<logger> clone(std::string logger_name);

    protected:
        std::string name_;
        std::vector<sink_ptr> sinks_;
        turbo::tlog::level_t level_{level::info};
        turbo::tlog::level_t flush_level_{level::off};
        err_handler custom_err_handler_{nullptr};
        details::backtracer tracer_;

        // common implementation for after templated public api has been resolved
        template<typename... Args>
        void log_(source_loc loc, level::level_enum lvl, std::string_view fmt, Args &&... args) {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled) {
                return;
            }
            TLOG_TRY {
                memory_buf_t buf;
                turbo::vformat_to(turbo::appender(buf), fmt, turbo::make_format_args(std::forward<Args>(args)...));
                details::log_msg log_msg(loc, name_, lvl, std::string_view(buf.data(), buf.size()));
                log_it_(log_msg, log_enabled, traceback_enabled);
            }
            TLOG_LOGGER_CATCH(loc)
        }

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT
        template<typename... Args>
        void log_(source_loc loc, level::level_enum lvl, wstring_view_t fmt, Args &&... args)
        {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled)
            {
                return;
            }
            TLOG_TRY
            {
                // format to wmemory_buffer and convert to utf8
                wmemory_buf_t wbuf;
#    ifdef TLOG_USE_STD_FORMAT
                turbo::vformat_to(
                    std::back_inserter(wbuf), fmt, turbo::make_format_args<turbo::wformat_context>(std::forward<Args>(args)...));
#    else
                turbo::vformat_to(std::back_inserter(wbuf), fmt, turbo::make_format_args<turbo::wformat_context>(std::forward<Args>(args)...));
#    endif

                memory_buf_t buf;
                details::os::wstr_to_utf8buf(wstring_view_t(wbuf.data(), wbuf.size()), buf);
                details::log_msg log_msg(loc, name_, lvl, std::string_view(buf.data(), buf.size()));
                log_it_(log_msg, log_enabled, traceback_enabled);
            }
            TLOG_LOGGER_CATCH(loc)
        }

        // T can be statically converted to wstring_view, and no formatting needed.
        template<class T, typename std::enable_if<std::is_convertible<const T &, turbo::tlog::wstring_view_t>::value, int>::type = 0>
        void log_(source_loc loc, level::level_enum lvl, const T &msg)
        {
            bool log_enabled = should_log(lvl);
            bool traceback_enabled = tracer_.enabled();
            if (!log_enabled && !traceback_enabled)
            {
                return;
            }
            TLOG_TRY
            {
                memory_buf_t buf;
                details::os::wstr_to_utf8buf(msg, buf);
                details::log_msg log_msg(loc, name_, lvl, std::string_view(buf.data(), buf.size()));
                log_it_(log_msg, log_enabled, traceback_enabled);
            }
            TLOG_LOGGER_CATCH(loc)
        }

#endif // TLOG_WCHAR_TO_UTF8_SUPPORT

        // log the given message (if the given log level is high enough),
        // and save backtrace (if backtrace is enabled).
        void log_it_(const details::log_msg &log_msg, bool log_enabled, bool traceback_enabled);

        virtual void sink_it_(const details::log_msg &msg);

        virtual void flush_();

        void dump_backtrace_();

        bool should_flush_(const details::log_msg &msg);

        // handle errors during logging.
        // default handler prints the error to stderr at max rate of 1 message/sec.
        void err_handler_(const std::string &msg);
    };

    void swap(logger &a, logger &b);

} // namespace turbo::tlog

