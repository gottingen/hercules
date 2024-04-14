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

#include "dist_sink.h"
#include "turbo/log/details/null_mutex.h"
#include "turbo/log/details/log_msg.h"

#include <cstdio>
#include <mutex>
#include <string>
#include <chrono>

// Duplicate message removal sink.
// Skip the message if previous one is identical and less than "max_skip_duration" have passed
//
// Example:
//
//     #include <turbo/log/sinks/dup_filter_sink.h>
//
//     int main() {
//         auto dup_filter = std::make_shared<dup_filter_sink_st>(std::chrono::seconds(5));
//         dup_filter->add_sink(std::make_shared<stdout_color_sink_mt>());
//         turbo::tlog::logger l("logger", dup_filter);
//         l.info("Hello");
//         l.info("Hello");
//         l.info("Hello");
//         l.info("Different Hello");
//     }
//
// Will produce:
//       [2019-06-25 17:50:56.511] [logger] [info] Hello
//       [2019-06-25 17:50:56.512] [logger] [info] Skipped 3 duplicate messages..
//       [2019-06-25 17:50:56.512] [logger] [info] Different Hello

namespace turbo::tlog {
    namespace sinks {
        template<typename Mutex>
        class dup_filter_sink : public dist_sink<Mutex> {
        public:
            explicit dup_filter_sink(turbo::Duration max_skip_duration)
                    : max_skip_duration_{max_skip_duration} {}

        protected:
            turbo::Duration max_skip_duration_;
            turbo::Time last_msg_time_;
            std::string last_msg_payload_;
            size_t skip_counter_ = 0;

            void sink_it_(const details::log_msg &msg) override {
                bool filtered = filter_(msg);
                if (!filtered) {
                    skip_counter_ += 1;
                    return;
                }

                // log the "skipped.." message
                if (skip_counter_ > 0) {
                    char buf[64];
                    auto msg_size = ::snprintf(buf, sizeof(buf), "Skipped %u duplicate messages..",
                                               static_cast<unsigned>(skip_counter_));
                    if (msg_size > 0 && static_cast<size_t>(msg_size) < sizeof(buf)) {
                        details::log_msg skipped_msg{msg.logger_name, level::info,
                                                     std::string_view{buf, static_cast<size_t>(msg_size)}};
                        dist_sink<Mutex>::sink_it_(skipped_msg);
                    }
                }

                // log current message
                dist_sink<Mutex>::sink_it_(msg);
                last_msg_time_ = msg.time;
                skip_counter_ = 0;
                last_msg_payload_.assign(msg.payload.data(), msg.payload.data() + msg.payload.size());
            }

            // return whether the log msg should be displayed (true) or skipped (false)
            bool filter_(const details::log_msg &msg) {
                auto filter_duration = msg.time - last_msg_time_;
                return (filter_duration > max_skip_duration_) || (msg.payload != last_msg_payload_);
            }
        };

        using dup_filter_sink_mt = dup_filter_sink<std::mutex>;
        using dup_filter_sink_st = dup_filter_sink<details::null_mutex>;

    } // namespace sinks
} // namespace turbo::tlog
