// Copyright 2023 The Turbo Authors.
//
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

#include "turbo/log/details/null_mutex.h"
#include "turbo/log/sinks/base_sink.h"
#include <chrono>
#include <mutex>
#include <thread>

namespace turbo::tlog {
    namespace sinks {

        template<class Mutex>
        class test_sink : public base_sink<Mutex> {
            const size_t lines_to_save = 100;

        public:
            size_t msg_counter() {
                std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
                return msg_counter_;
            }

            size_t flush_counter() {
                std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
                return flush_counter_;
            }

            void set_delay(std::chrono::milliseconds delay) {
                std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
                delay_ = delay;
            }

            // return last output without the eol
            std::vector<std::string> lines() {
                std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
                return lines_;
            }

        protected:
            void sink_it_(const details::log_msg &msg) override {
                memory_buf_t formatted;
                base_sink<Mutex>::formatter_->format(msg, formatted);
                // save the line without the eol
                auto eol_len = strlen(details::os::default_eol);
                if (lines_.size() < lines_to_save) {
                    lines_.emplace_back(formatted.begin(), formatted.end() - eol_len);
                }
                msg_counter_++;
                std::this_thread::sleep_for(delay_);
            }

            void flush_() override {
                flush_counter_++;
            }

            size_t msg_counter_{0};
            size_t flush_counter_{0};
            std::chrono::milliseconds delay_{std::chrono::milliseconds::zero()};
            std::vector<std::string> lines_;
        };

        using test_sink_mt = test_sink<std::mutex>;
        using test_sink_st = test_sink<details::null_mutex>;

    } // namespace sinks
} // namespace spdlog
