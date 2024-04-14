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

#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/null_mutex.h"
#include "turbo/log/details/synchronous_factory.h"

#include <array>
#include <string>
#include <syslog.h>

namespace turbo::tlog {
    namespace sinks {
/**
 * Sink that write to syslog using the `syscall()` library call.
 */
        template<typename Mutex>
        class syslog_sink : public base_sink<Mutex> {

        public:
            syslog_sink(std::string ident, int syslog_option, int syslog_facility, bool enable_formatting)
                    : enable_formatting_{enable_formatting},
                      syslog_levels_{{/* turbo::tlog::level::trace      */ LOG_DEBUG,
                                             /* turbo::tlog::level::debug      */ LOG_DEBUG,
                                             /* turbo::tlog::level::info       */ LOG_INFO,
                                             /* turbo::tlog::level::warn       */ LOG_WARNING,
                                             /* turbo::tlog::level::err        */ LOG_ERR,
                                             /* turbo::tlog::level::critical   */ LOG_CRIT,
                                             /* turbo::tlog::level::off        */ LOG_INFO}}, ident_{std::move(ident)} {
                // set ident to be program name if empty
                ::openlog(ident_.empty() ? nullptr : ident_.c_str(), syslog_option, syslog_facility);
            }

            ~syslog_sink() override {
                ::closelog();
            }

            syslog_sink(const syslog_sink &) = delete;

            syslog_sink &operator=(const syslog_sink &) = delete;

        protected:
            void sink_it_(const details::log_msg &msg) override {
                std::string_view payload;
                memory_buf_t formatted;
                if (enable_formatting_) {
                    base_sink<Mutex>::formatter_->format(msg, formatted);
                    payload = std::string_view(formatted.data(), formatted.size());
                } else {
                    payload = msg.payload;
                }

                size_t length = payload.size();
                // limit to max int
                if (length > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    length = static_cast<size_t>(std::numeric_limits<int>::max());
                }

                ::syslog(syslog_prio_from_level(msg), "%.*s", static_cast<int>(length), payload.data());
            }

            void flush_() override {}

            bool enable_formatting_ = false;

        private:
            using levels_array = std::array<int, 7>;
            levels_array syslog_levels_;
            // must store the ident because the man says openlog might use the pointer as
            // is and not a string copy
            const std::string ident_;

            //
            // Simply maps spdlog's log level to syslog priority level.
            //
            int syslog_prio_from_level(const details::log_msg &msg) const {
                return syslog_levels_.at(static_cast<levels_array::size_type>(msg.level));
            }
        };

        using syslog_sink_mt = syslog_sink<std::mutex>;
        using syslog_sink_st = syslog_sink<details::null_mutex>;
    } // namespace sinks

// Create and register a syslog logger
    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    syslog_logger_mt(const std::string &logger_name, const std::string &syslog_ident = "", int syslog_option = 0,
                     int syslog_facility = LOG_USER, bool enable_formatting = false) {
        return Factory::template create<sinks::syslog_sink_mt>(logger_name, syslog_ident, syslog_option,
                                                               syslog_facility, enable_formatting);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger>
    syslog_logger_st(const std::string &logger_name, const std::string &syslog_ident = "", int syslog_option = 0,
                     int syslog_facility = LOG_USER, bool enable_formatting = false) {
        return Factory::template create<sinks::syslog_sink_st>(logger_name, syslog_ident, syslog_option,
                                                               syslog_facility, enable_formatting);
    }
} // namespace turbo::tlog
