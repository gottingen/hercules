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

#include <turbo/log/details/log_msg_buffer.h>
#include <turbo/log/details/circular_q.h>

#include <atomic>
#include <mutex>
#include <functional>

// Store log messages in circular buffer.
// Useful for storing debug data in case of error/warning happens.

namespace turbo::tlog {
    namespace details {
        class TURBO_DLL backtracer {
            mutable std::mutex mutex_;
            std::atomic<bool> enabled_{false};
            circular_q<log_msg_buffer> messages_;

        public:
            backtracer() = default;

            backtracer(const backtracer &other);

            backtracer(backtracer &&other) noexcept;

            backtracer &operator=(backtracer other);

            void enable(size_t size);

            void disable();

            bool enabled() const;

            void push_back(const log_msg &msg);

            // pop all items in the q and apply the given fun on each of them.
            void foreach_pop(std::function<void(const details::log_msg &)> fun);
        };

    } // namespace details
} // namespace turbo::tlog

