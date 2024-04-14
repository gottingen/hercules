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

#include "turbo/log/logger.h"
#include <turbo/log/sinks/sink.h>
#include <turbo/log/details/backtracer.h>
#include <turbo/log/pattern_formatter.h>

#include <cstdio>

namespace turbo::tlog {

// public methods
    logger::logger(const logger &other)
            : name_(other.name_), sinks_(other.sinks_), level_(other.level_.load(std::memory_order_relaxed)),
              flush_level_(other.flush_level_.load(std::memory_order_relaxed)),
              custom_err_handler_(other.custom_err_handler_), tracer_(other.tracer_) {}

    logger::logger(logger &&other) noexcept: name_(std::move(other.name_)),
                                             sinks_(std::move(other.sinks_)),
                                             level_(other.level_.load(std::memory_order_relaxed)),
                                             flush_level_(other.flush_level_.load(std::memory_order_relaxed)),
                                             custom_err_handler_(std::move(other.custom_err_handler_)),
                                             tracer_(std::move(other.tracer_)) {}

    logger &logger::operator=(logger other) noexcept {
        this->swap(other);
        return *this;
    }

    void logger::swap(turbo::tlog::logger &other) noexcept {
        name_.swap(other.name_);
        sinks_.swap(other.sinks_);

        // swap level_
        auto other_level = other.level_.load();
        auto my_level = level_.exchange(other_level);
        other.level_.store(my_level);

        // swap flush level_
        other_level = other.flush_level_.load();
        my_level = flush_level_.exchange(other_level);
        other.flush_level_.store(my_level);

        custom_err_handler_.swap(other.custom_err_handler_);
        std::swap(tracer_, other.tracer_);
    }

    void swap(logger &a, logger &b) {
        a.swap(b);
    }

    void logger::set_level(level::level_enum log_level) {
        level_.store(log_level);
    }

    level::level_enum logger::level() const {
        return static_cast<level::level_enum>(level_.load(std::memory_order_relaxed));
    }

    const std::string &logger::name() const {
        return name_;
    }

    // set formatting for the sinks in this logger.
    // each sink will get a separate instance of the formatter object.
    void logger::set_formatter(std::unique_ptr<formatter> f) {
        for (auto it = sinks_.begin(); it != sinks_.end(); ++it) {
            if (std::next(it) == sinks_.end()) {
                // last element - we can be move it.
                (*it)->set_formatter(std::move(f));
                break; // to prevent clang-tidy warning
            } else {
                (*it)->set_formatter(f->clone());
            }
        }
    }

    void logger::set_pattern(std::string pattern, pattern_time_type time_type) {
        auto new_formatter = details::make_unique<pattern_formatter>(std::move(pattern), time_type);
        set_formatter(std::move(new_formatter));
    }

    // create new backtrace sink and move to it all our child sinks
    void logger::enable_backtrace(size_t n_messages) {
        tracer_.enable(n_messages);
    }

    // restore orig sinks and level and delete the backtrace sink
    void logger::disable_backtrace() {
        tracer_.disable();
    }

    void logger::dump_backtrace() {
        dump_backtrace_();
    }

    // flush functions
    void logger::flush() {
        flush_();
    }

    void logger::flush_on(level::level_enum log_level) {
        flush_level_.store(log_level);
    }

    level::level_enum logger::flush_level() const {
        return static_cast<level::level_enum>(flush_level_.load(std::memory_order_relaxed));
    }

// sinks
    const std::vector<sink_ptr> &logger::sinks() const {
        return sinks_;
    }

    std::vector<sink_ptr> &logger::sinks() {
        return sinks_;
    }

// error handler
    void logger::set_error_handler(err_handler handler) {
        custom_err_handler_ = std::move(handler);
    }

// create new logger with same sinks and configuration.
    std::shared_ptr<logger> logger::clone(std::string logger_name) {
        auto cloned = std::make_shared<logger>(*this);
        cloned->name_ = std::move(logger_name);
        return cloned;
    }

// protected methods
    void logger::log_it_(const turbo::tlog::details::log_msg &log_msg, bool log_enabled, bool traceback_enabled) {
        if (log_enabled) {
            sink_it_(log_msg);
        }
        if (traceback_enabled) {
            tracer_.push_back(log_msg);
        }
    }

    void logger::sink_it_(const details::log_msg &msg) {
        for (auto &sink: sinks_) {
            if (sink->should_log(msg.level)) {
                TLOG_TRY {
                    sink->log(msg);
                }
                TLOG_LOGGER_CATCH(msg.source)
            }
        }

        if (should_flush_(msg)) {
            flush_();
        }
    }

    void logger::flush_() {
        for (auto &sink: sinks_) {
            TLOG_TRY {
                sink->flush();
            }
            TLOG_LOGGER_CATCH(source_loc())
        }
    }

    void logger::dump_backtrace_() {
        using details::log_msg;
        if (tracer_.enabled()) {
            sink_it_(log_msg{name(), level::info, "****************** Backtrace Start ******************"});
            tracer_.foreach_pop([this](const log_msg &msg) { this->sink_it_(msg); });
            sink_it_(log_msg{name(), level::info, "****************** Backtrace End ********************"});
        }
    }

    bool logger::should_flush_(const details::log_msg &msg) {
        auto flush_level = flush_level_.load(std::memory_order_relaxed);
        return (msg.level >= flush_level) && (msg.level != level::off);
    }

    void logger::err_handler_(const std::string &msg) {
        if (custom_err_handler_) {
            custom_err_handler_(msg);
        } else {
            using std::chrono::system_clock;
            static std::mutex mutex;
            static std::chrono::system_clock::time_point last_report_time;
            static size_t err_counter = 0;
            std::lock_guard<std::mutex> lk{mutex};
            auto now = system_clock::now();
            err_counter++;
            if (now - last_report_time < std::chrono::seconds(1)) {
                return;
            }
            last_report_time = now;
            auto tm_time = details::os::localtime(system_clock::to_time_t(now));
            char date_buf[64];
            std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%d %H:%M:%S", &tm_time);
#if defined(USING_R) && defined(R_R_H) // if in R environment
            REprintf("[*** LOG ERROR #%04zu ***] [%s] [%s] {%s}\n", err_counter, date_buf, name().c_str(), msg.c_str());
#else
            std::fprintf(stderr, "[*** LOG ERROR #%04zu ***] [%s] [%s] {%s}\n", err_counter, date_buf, name().c_str(),
                         msg.c_str());
#endif
        }
    }
} // namespace turbo::tlog
