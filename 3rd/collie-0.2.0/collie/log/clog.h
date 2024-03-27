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

#ifndef COLLIE_CLOG_H_
#define COLLIE_CLOG_H_

#pragma once

#include <collie/log/common.h>
#include <collie/log/details/registry.h>
#include <collie/log/details/synchronous_factory.h>
#include <collie/log/logger.h>
#include <collie/log/version.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace clog {

    using default_factory = synchronous_factory;

    // Create and register a logger with a templated sink type
    // The logger's level, formatter and flush level will be set according the
    // global settings.
    //
    // Example:
    //   clog::create<daily_file_sink_st>("logger_name", "dailylog_filename", 11, 59);
    template<typename Sink, typename... SinkArgs>
    inline std::shared_ptr<clog::logger> create(std::string logger_name, SinkArgs &&...sink_args) {
        return default_factory::create<Sink>(std::move(logger_name),
                                             std::forward<SinkArgs>(sink_args)...);
    }

    // Initialize and register a logger,
    // formatter and flush level will be set according the global settings.
    //
    // Useful for initializing manually created loggers with the global settings.
    //
    // Example:
    //   auto mylogger = std::make_shared<clog::logger>("mylogger", ...);
    //   clog::initialize_logger(mylogger);
    void initialize_logger(std::shared_ptr<logger> logger);

    // Return an existing logger or nullptr if a logger with such name doesn't
    // exist.
    // example: clog::get("my_logger")->info("hello {}", "world");
    std::shared_ptr<logger> get(const std::string &name);

    // Set global formatter. Each sink in each logger will get a clone of this object
    void set_formatter(std::unique_ptr<clog::formatter> formatter);

    // Set global format string.
    // example: clog::set_pattern("%Y-%m-%d %H:%M:%S.%e %l : %v");
    void set_pattern(std::string pattern,
                     pattern_time_type time_type = pattern_time_type::local);

    // enable global backtrace support
    void enable_backtrace(size_t n_messages);

    // disable global backtrace support
    void disable_backtrace();

    // call dump backtrace on default logger
    void dump_backtrace();

    // Get global logging level
    level::level_enum get_level();

    // Set global logging level
    void set_level(level::level_enum log_level);

    // Determine whether the default logger should log messages with a certain level
    bool should_log(level::level_enum lvl);

    // Set global flush level
    void flush_on(level::level_enum log_level);

    // Start/Restart a periodic flusher thread
    // Warning: Use only if all your loggers are thread safe!
    template<typename Rep, typename Period>
    inline void flush_every(std::chrono::duration<Rep, Period> interval) {
        details::registry::instance().flush_every(interval);
    }

    // Set global error handler
    void set_error_handler(void (*handler)(const std::string &msg));

    // Register the given logger with the given name
    void register_logger(std::shared_ptr<logger> logger);

    // Apply a user defined function on all registered loggers
    // Example:
    // clog::apply_all([&](std::shared_ptr<clog::logger> l) {l->flush();});
    void apply_all(const std::function<void(std::shared_ptr<logger>)> &fun);

    // Drop the reference to the given logger
    void drop(const std::string &name);

    // Drop all references from the registry
    void drop_all();

    // stop any running threads started by clog and clean registry loggers
    void shutdown();

    // Automatic registration of loggers when using clog::create() or clog::create_async
    void set_automatic_registration(bool automatic_registration);

    // API for using default logger (stdout_color_mt),
    // e.g: clog::info("Message {}", 1);
    //
    // The default logger object can be accessed using the clog::default_logger():
    // For example, to add another sink to it:
    // clog::default_logger()->sinks().push_back(some_sink);
    //
    // The default logger can replaced using clog::set_default_logger(new_logger).
    // For example, to replace it with a file logger.
    //
    // IMPORTANT:
    // The default API is thread safe (for _mt loggers), but:
    // set_default_logger() *should not* be used concurrently with the default API.
    // e.g do not call set_default_logger() from one thread while calling clog::info() from another.

    std::shared_ptr<clog::logger> default_logger();

    clog::logger *default_logger_raw();

    void set_default_logger(std::shared_ptr<clog::logger> default_logger);

    // Initialize logger level based on environment configs.
    //
    // Useful for applying CLOG_LEVEL to manually created loggers.
    //
    // Example:
    //   auto mylogger = std::make_shared<clog::logger>("mylogger", ...);
    //   clog::apply_logger_env_levels(mylogger);
    void apply_logger_env_levels(std::shared_ptr<logger> logger);

    template<typename... Args>
    inline void log(source_loc source,
                    level::level_enum lvl,
                    format_string_t<Args...> fmt,
                    Args &&...args) {
        default_logger_raw()->log(source, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void log(level::level_enum lvl, format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void trace(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void debug(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void info(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void warn(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void error(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void critical(format_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->critical(fmt, std::forward<Args>(args)...);
    }

    template<typename T>
    inline void log(source_loc source, level::level_enum lvl, const T &msg) {
        default_logger_raw()->log(source, lvl, msg);
    }

    template<typename T>
    inline void log(level::level_enum lvl, const T &msg) {
        default_logger_raw()->log(lvl, msg);
    }

#ifdef CLOG_WCHAR_TO_UTF8_SUPPORT
    template <typename... Args>
    inline void log(source_loc source,
                    level::level_enum lvl,
                    wformat_string_t<Args...> fmt,
                    Args &&...args) {
        default_logger_raw()->log(source, lvl, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void log(level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void trace(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->trace(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void debug(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->debug(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void info(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->info(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void warn(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->warn(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void error(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->error(fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void critical(wformat_string_t<Args...> fmt, Args &&...args) {
        default_logger_raw()->critical(fmt, std::forward<Args>(args)...);
    }
#endif

    template<typename T>
    inline void trace(const T &msg) {
        default_logger_raw()->trace(msg);
    }

    template<typename T>
    inline void debug(const T &msg) {
        default_logger_raw()->debug(msg);
    }

    template<typename T>
    inline void info(const T &msg) {
        default_logger_raw()->info(msg);
    }

    template<typename T>
    inline void warn(const T &msg) {
        default_logger_raw()->warn(msg);
    }

    template<typename T>
    inline void error(const T &msg) {
        default_logger_raw()->error(msg);
    }

    template<typename T>
    inline void critical(const T &msg) {
        default_logger_raw()->critical(msg);
    }

}  // namespace clog

//
// enable/disable log calls at compile time according to global level.
//
// define CLOG_ACTIVE_LEVEL to one of those (before including clog.h):
// CLOG_LEVEL_TRACE,
// CLOG_LEVEL_DEBUG,
// CLOG_LEVEL_INFO,
// CLOG_LEVEL_WARN,
// CLOG_LEVEL_ERROR,
// CLOG_LEVEL_CRITICAL,
// CLOG_LEVEL_OFF
//

#ifndef CLOG_NO_SOURCE_LOC
#define CLOG_LOGGER_CALL(logger, level, ...) \
        (logger)->log(clog::source_loc{__FILE__, __LINE__, CLOG_FUNCTION}, level, __VA_ARGS__)
#else
#define CLOG_LOGGER_CALL(logger, level, ...) \
        (logger)->log(clog::source_loc{}, level, __VA_ARGS__)
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_TRACE
#define CLOG_LOGGER_TRACE(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::trace, __VA_ARGS__)
#define CLOG_TRACE(...) CLOG_LOGGER_TRACE(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_TRACE(logger, ...) (void)0
#define CLOG_TRACE(...) (void)0
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_DEBUG
#define CLOG_LOGGER_DEBUG(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::debug, __VA_ARGS__)
#define CLOG_DEBUG(...) CLOG_LOGGER_DEBUG(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_DEBUG(logger, ...) (void)0
#define CLOG_DEBUG(...) (void)0
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_INFO
#define CLOG_LOGGER_INFO(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::info, __VA_ARGS__)
#define CLOG_INFO(...) CLOG_LOGGER_INFO(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_INFO(logger, ...) (void)0
#define CLOG_INFO(...) (void)0
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_WARN
#define CLOG_LOGGER_WARN(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::warn, __VA_ARGS__)
#define CLOG_WARN(...) CLOG_LOGGER_WARN(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_WARN(logger, ...) (void)0
#define CLOG_WARN(...) (void)0
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_ERROR
#define CLOG_LOGGER_ERROR(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::err, __VA_ARGS__)
#define CLOG_ERROR(...) CLOG_LOGGER_ERROR(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_ERROR(logger, ...) (void)0
#define CLOG_ERROR(...) (void)0
#endif

#if CLOG_ACTIVE_LEVEL <= CLOG_LEVEL_CRITICAL
#define CLOG_LOGGER_CRITICAL(logger, ...) \
        CLOG_LOGGER_CALL(logger, clog::level::critical, __VA_ARGS__)
#define CLOG_CRITICAL(...) CLOG_LOGGER_CRITICAL(clog::default_logger_raw(), __VA_ARGS__)
#else
#define CLOG_LOGGER_CRITICAL(logger, ...) (void)0
#define CLOG_CRITICAL(...) (void)0
#endif

#include <collie/log/clog-inl.h>

#endif  // COLLIE_CLOG_H_
