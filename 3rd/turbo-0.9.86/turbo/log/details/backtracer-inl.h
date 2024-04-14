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

#include <turbo/log/details/backtracer.h>

namespace turbo::tlog {
    namespace details {
        backtracer::backtracer(const backtracer &other) {
            std::lock_guard<std::mutex> lock(other.mutex_);
            enabled_ = other.enabled();
            messages_ = other.messages_;
        }

        backtracer::backtracer(backtracer &&other) noexcept {
            std::lock_guard<std::mutex> lock(other.mutex_);
            enabled_ = other.enabled();
            messages_ = std::move(other.messages_);
        }

        backtracer &backtracer::operator=(backtracer other) {
            std::lock_guard<std::mutex> lock(mutex_);
            enabled_ = other.enabled();
            messages_ = std::move(other.messages_);
            return *this;
        }

        void backtracer::enable(size_t size) {
            std::lock_guard<std::mutex> lock{mutex_};
            enabled_.store(true, std::memory_order_relaxed);
            messages_ = circular_q<log_msg_buffer>{size};
        }

        void backtracer::disable() {
            std::lock_guard<std::mutex> lock{mutex_};
            enabled_.store(false, std::memory_order_relaxed);
        }

        bool backtracer::enabled() const {
            return enabled_.load(std::memory_order_relaxed);
        }

        void backtracer::push_back(const log_msg &msg) {
            std::lock_guard<std::mutex> lock{mutex_};
            messages_.push_back(log_msg_buffer{msg});
        }

// pop all items in the q and apply the given fun on each of them.
        void backtracer::foreach_pop(std::function<void(const details::log_msg &)> fun) {
            std::lock_guard<std::mutex> lock{mutex_};
            while (!messages_.empty()) {
                auto &front_msg = messages_.front();
                fun(front_msg);
                messages_.pop_front();
            }
        }
    } // namespace details
} // namespace turbo::tlog
