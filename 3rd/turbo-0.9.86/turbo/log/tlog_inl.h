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

#include <turbo/log/tlog.h>
#include "turbo/log/common.h"
#include <turbo/log/pattern_formatter.h>

namespace turbo::tlog {

    void initialize_logger(std::shared_ptr<logger> logger) {
        details::registry::instance().initialize_logger(std::move(logger));
    }

    std::shared_ptr<logger> get(const std::string &name) {
        return details::registry::instance().get(name);
    }

    void set_formatter(std::unique_ptr<turbo::tlog::formatter> formatter) {
        details::registry::instance().set_formatter(std::move(formatter));
    }

    void set_pattern(std::string pattern, pattern_time_type time_type) {
        set_formatter(std::unique_ptr<turbo::tlog::formatter>(new pattern_formatter(std::move(pattern), time_type)));
    }

    void enable_backtrace(size_t n_messages) {
        details::registry::instance().enable_backtrace(n_messages);
    }

    void disable_backtrace() {
        details::registry::instance().disable_backtrace();
    }

    void dump_backtrace() {
        default_logger_raw()->dump_backtrace();
    }

    level::level_enum get_level() {
        return default_logger_raw()->level();
    }

    bool should_log(level::level_enum log_level) {
        return default_logger_raw()->should_log(log_level);
    }

    void set_level(level::level_enum log_level) {
        details::registry::instance().set_level(log_level);
    }

    void flush_on(level::level_enum log_level) {
        details::registry::instance().flush_on(log_level);
    }

    void set_error_handler(void (*handler)(const std::string &msg)) {
        details::registry::instance().set_error_handler(handler);
    }

    void register_logger(std::shared_ptr<logger> logger) {
        details::registry::instance().register_logger(std::move(logger));
    }

    void apply_all(const std::function<void(std::shared_ptr<logger>)> &fun) {
        details::registry::instance().apply_all(fun);
    }

    void drop(const std::string &name) {
        details::registry::instance().drop(name);
    }

    void drop_all() {
        details::registry::instance().drop_all();
    }

    void shutdown() {
        details::registry::instance().shutdown();
    }

    void set_automatic_registration(bool automatic_registration) {
        details::registry::instance().set_automatic_registration(automatic_registration);
    }

    std::shared_ptr<turbo::tlog::logger> default_logger() {
        return details::registry::instance().default_logger();
    }

    turbo::tlog::logger *default_logger_raw() {
        return details::registry::instance().get_default_raw();
    }

    void set_default_logger(std::shared_ptr<turbo::tlog::logger> default_logger) {
        details::registry::instance().set_default_logger(std::move(default_logger));
    }

} // namespace turbo::tlog
