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

#include <collie/log/common.h>
#include <collie/log/details/periodic_worker.h>
#include <collie/log/logger.h>
#include <collie/log/pattern_formatter.h>

#ifndef CLOG_DISABLE_DEFAULT_LOGGER
    // support for the default stdout color logger
    #ifdef _WIN32
        #include <collie/log/sinks/wincolor_sink.h>
    #else
        #include <collie/log/sinks/ansicolor_sink.h>
    #endif
#endif  // CLOG_DISABLE_DEFAULT_LOGGER

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace clog {
namespace details {

inline registry::registry()
    : formatter_(new pattern_formatter()) {
#ifndef CLOG_DISABLE_DEFAULT_LOGGER
    // create default logger (ansicolor_stdout_sink_mt or wincolor_stdout_sink_mt in windows).
    #ifdef _WIN32
    auto color_sink = std::make_shared<sinks::wincolor_stdout_sink_mt>();
    #else
    auto color_sink = std::make_shared<sinks::ansicolor_stdout_sink_mt>();
    #endif

    const char *default_logger_name = "";
    default_logger_ = std::make_shared<clog::logger>(default_logger_name, std::move(color_sink));
    loggers_[default_logger_name] = default_logger_;

#endif  // CLOG_DISABLE_DEFAULT_LOGGER
}

inline registry::~registry() = default;

inline void registry::register_logger(std::shared_ptr<logger> new_logger) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    register_logger_(std::move(new_logger));
}

inline void registry::initialize_logger(std::shared_ptr<logger> new_logger) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    new_logger->set_formatter(formatter_->clone());

    if (err_handler_) {
        new_logger->set_error_handler(err_handler_);
    }

    // set new level according to previously configured level or default level
    auto it = log_levels_.find(new_logger->name());
    auto new_level = it != log_levels_.end() ? it->second : global_log_level_;
    new_logger->set_level(new_level);

    new_logger->flush_on(flush_level_);

    if (backtrace_n_messages_ > 0) {
        new_logger->enable_backtrace(backtrace_n_messages_);
    }

    if (automatic_registration_) {
        register_logger_(std::move(new_logger));
    }
}

inline std::shared_ptr<logger> registry::get(const std::string &logger_name) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    auto found = loggers_.find(logger_name);
    return found == loggers_.end() ? nullptr : found->second;
}

inline std::shared_ptr<logger> registry::default_logger() {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    return default_logger_;
}

// Return raw ptr to the default logger.
// To be used directly by the clog default api (e.g. clog::info)
// This make the default API faster, but cannot be used concurrently with set_default_logger().
// e.g do not call set_default_logger() from one thread while calling clog::info() from another.
inline logger *registry::get_default_raw() { return default_logger_.get(); }

// set default logger.
// default logger is stored in default_logger_ (for faster retrieval) and in the loggers_ map.
inline void registry::set_default_logger(std::shared_ptr<logger> new_default_logger) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    // remove previous default logger from the map
    if (default_logger_ != nullptr) {
        loggers_.erase(default_logger_->name());
    }
    if (new_default_logger != nullptr) {
        loggers_[new_default_logger->name()] = new_default_logger;
    }
    default_logger_ = std::move(new_default_logger);
}

inline void registry::set_tp(std::shared_ptr<thread_pool> tp) {
    std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
    tp_ = std::move(tp);
}

inline std::shared_ptr<thread_pool> registry::get_tp() {
    std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
    return tp_;
}

// Set global formatter. Each sink in each logger will get a clone of this object
inline void registry::set_formatter(std::unique_ptr<formatter> formatter) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    formatter_ = std::move(formatter);
    for (auto &l : loggers_) {
        l.second->set_formatter(formatter_->clone());
    }
}

inline void registry::enable_backtrace(size_t n_messages) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    backtrace_n_messages_ = n_messages;

    for (auto &l : loggers_) {
        l.second->enable_backtrace(n_messages);
    }
}

inline void registry::disable_backtrace() {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    backtrace_n_messages_ = 0;
    for (auto &l : loggers_) {
        l.second->disable_backtrace();
    }
}

inline void registry::set_level(level::level_enum log_level) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    for (auto &l : loggers_) {
        l.second->set_level(log_level);
    }
    global_log_level_ = log_level;
}

inline void registry::flush_on(level::level_enum log_level) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    for (auto &l : loggers_) {
        l.second->flush_on(log_level);
    }
    flush_level_ = log_level;
}

inline void registry::set_error_handler(err_handler handler) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    for (auto &l : loggers_) {
        l.second->set_error_handler(handler);
    }
    err_handler_ = std::move(handler);
}

inline void registry::apply_all(
    const std::function<void(const std::shared_ptr<logger>)> &fun) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    for (auto &l : loggers_) {
        fun(l.second);
    }
}

inline void registry::flush_all() {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    for (auto &l : loggers_) {
        l.second->flush();
    }
}

inline void registry::drop(const std::string &logger_name) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    auto is_default_logger = default_logger_ && default_logger_->name() == logger_name;
    loggers_.erase(logger_name);
    if (is_default_logger) {
        default_logger_.reset();
    }
}

inline void registry::drop_all() {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    loggers_.clear();
    default_logger_.reset();
}

// clean all resources and threads started by the registry
inline void registry::shutdown() {
    {
        std::lock_guard<std::mutex> lock(flusher_mutex_);
        periodic_flusher_.reset();
    }

    drop_all();

    {
        std::lock_guard<std::recursive_mutex> lock(tp_mutex_);
        tp_.reset();
    }
}

inline std::recursive_mutex &registry::tp_mutex() { return tp_mutex_; }

inline void registry::set_automatic_registration(bool automatic_registration) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    automatic_registration_ = automatic_registration;
}

inline void registry::set_levels(log_levels levels, level::level_enum *global_level) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    log_levels_ = std::move(levels);
    auto global_level_requested = global_level != nullptr;
    global_log_level_ = global_level_requested ? *global_level : global_log_level_;

    for (auto &logger : loggers_) {
        auto logger_entry = log_levels_.find(logger.first);
        if (logger_entry != log_levels_.end()) {
            logger.second->set_level(logger_entry->second);
        } else if (global_level_requested) {
            logger.second->set_level(*global_level);
        }
    }
}

inline registry &registry::instance() {
    static registry s_instance;
    return s_instance;
}

inline void registry::apply_logger_env_levels(std::shared_ptr<logger> new_logger) {
    std::lock_guard<std::mutex> lock(logger_map_mutex_);
    auto it = log_levels_.find(new_logger->name());
    auto new_level = it != log_levels_.end() ? it->second : global_log_level_;
    new_logger->set_level(new_level);
}

inline void registry::throw_if_exists_(const std::string &logger_name) {
    if (loggers_.find(logger_name) != loggers_.end()) {
        throw_clog_ex("logger with name '" + logger_name + "' already exists");
    }
}

inline void registry::register_logger_(std::shared_ptr<logger> new_logger) {
    auto logger_name = new_logger->name();
    throw_if_exists_(logger_name);
    loggers_[logger_name] = std::move(new_logger);
}

}  // namespace details
}  // namespace clog
