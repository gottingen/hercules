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

// spdlog main header file.
// see example.cpp for usage example

#ifndef TURBO_LOG_TLOG_H_
#define TURBO_LOG_TLOG_H_

#pragma once

#include "turbo/log/common.h"
#include "turbo/log/details/registry.h"
#include "turbo/log/logger.h"
#include "turbo/log/details/synchronous_factory.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace turbo::tlog {

    using default_factory = synchronous_factory;

    // Create and register a logger with a templated sink type
    // The logger's level, formatter and flush level will be set according the
    // global settings.
    //
    // Example:
    //   turbo::tlog::create<daily_file_sink_st>("logger_name", "dailylog_filename", 11, 59);
    template<typename Sink, typename... SinkArgs>
    inline std::shared_ptr<turbo::tlog::logger> create(std::string logger_name, SinkArgs &&... sink_args) {
        return default_factory::create<Sink>(std::move(logger_name), std::forward<SinkArgs>(sink_args)...);
    }

    // Initialize and register a logger,
    // formatter and flush level will be set according the global settings.
    //
    // Useful for initializing manually created loggers with the global settings.
    //
    // Example:
    //   auto mylogger = std::make_shared<turbo::tlog::logger>("mylogger", ...);
    //   turbo::tlog::initialize_logger(mylogger);
    TURBO_DLL void initialize_logger(std::shared_ptr<logger> logger);

    // Return an existing logger or nullptr if a logger with such name doesn't
    // exist.
    // example: turbo::tlog::get("my_logger")->info("hello {}", "world");
    TURBO_DLL std::shared_ptr<logger> get(const std::string &name);

    // Set global formatter. Each sink in each logger will get a clone of this object
    TURBO_DLL void set_formatter(std::unique_ptr<turbo::tlog::formatter> formatter);

    // Set global format string.
    // example: turbo::tlog::set_pattern("%Y-%m-%d %H:%M:%S.%e %l : %v");
    TURBO_DLL void set_pattern(std::string pattern, pattern_time_type time_type = pattern_time_type::local);

    // enable global backtrace support
    TURBO_DLL void enable_backtrace(size_t n_messages);

    // disable global backtrace support
    TURBO_DLL void disable_backtrace();

    // call dump backtrace on default logger
    TURBO_DLL void dump_backtrace();

    // Get global logging level
    TURBO_DLL level::level_enum get_level();

    // Set global logging level
    TURBO_DLL void set_level(level::level_enum log_level);

    // Determine whether the default logger should log messages with a certain level
    TURBO_DLL bool should_log(level::level_enum lvl);

    // Set global flush level
    TURBO_DLL void flush_on(level::level_enum log_level);

    // Start/Restart a periodic flusher thread
    // Warning: Use only if all your loggers are thread safe!
    template<typename Rep, typename Period>
    inline void flush_every(std::chrono::duration<Rep, Period> interval) {
        details::registry::instance().flush_every(interval);
    }

    // Set global error handler
    TURBO_DLL void set_error_handler(void (*handler)(const std::string &msg));

    // Register the given logger with the given name
    TURBO_DLL void register_logger(std::shared_ptr<logger> logger);

    // Apply a user defined function on all registered loggers
    // Example:
    // turbo::tlog::apply_all([&](std::shared_ptr<turbo::tlog::logger> l) {l->flush();});
    TURBO_DLL void apply_all(const std::function<void(std::shared_ptr<logger>)> &fun);

    // Drop the reference to the given logger
    TURBO_DLL void drop(const std::string &name);

    // Drop all references from the registry
    TURBO_DLL void drop_all();

    // stop any running threads started by spdlog and clean registry loggers
    TURBO_DLL void shutdown();

    // Automatic registration of loggers when using turbo::tlog::create() or turbo::tlog::create_async
    TURBO_DLL void set_automatic_registration(bool automatic_registration);

    // API for using default logger (stdout_color_mt),
    // e.g: turbo::tlog::info("Message {}", 1);
    //
    // The default logger object can be accessed using the turbo::tlog::default_logger():
    // For example, to add another sink to it:
    // turbo::tlog::default_logger()->sinks().push_back(some_sink);
    //
    // The default logger can replaced using turbo::tlog::set_default_logger(new_logger).
    // For example, to replace it with a file logger.
    //
    // IMPORTANT:
    // The default API is thread safe (for _mt loggers), but:
    // set_default_logger() *should not* be used concurrently with the default API.
    // e.g do not call set_default_logger() from one thread while calling turbo::tlog::info() from another.

    TURBO_DLL std::shared_ptr<turbo::tlog::logger> default_logger();

    TURBO_DLL turbo::tlog::logger *default_logger_raw();

    TURBO_DLL void set_default_logger(std::shared_ptr<turbo::tlog::logger> default_logger);

    template<typename... Args>
    inline void log(source_loc source, level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->log(source, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void log(level::level_enum lvl, format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void trace(format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void debug(format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void info(format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void warn(format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void error(format_string_t<Args...> fmt, Args &&... args) {
        default_logger_raw()->error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void critical(format_string_t<Args...> fmt, Args &&... args) {
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

#ifdef TLOG_WCHAR_TO_UTF8_SUPPORT
    template<typename... Args>
    inline void log(source_loc source, level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->log(source, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void log(level::level_enum lvl, wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->log(source_loc{}, lvl, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void trace(wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->trace(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void debug(wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->debug(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void info(wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void warn(wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->warn(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void error(wformat_string_t<Args...> fmt, Args &&... args)
    {
        default_logger_raw()->error(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    inline void critical(wformat_string_t<Args...> fmt, Args &&... args)
    {
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

} // namespace turbo::tlog


#endif // TURBO_LOG_TLOG_H_
