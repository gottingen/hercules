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

#include <collie/log/details/circular_q.h>
#include <collie/log/details/log_msg_buffer.h>

#include <atomic>
#include <functional>
#include <mutex>

// Store log messages in circular buffer.
// Useful for storing debug data in case of error/warning happens.

namespace clog {
namespace details {
class  backtracer {
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
    bool empty() const;

    // pop all items in the q and apply the given fun on each of them.
    void foreach_pop(std::function<void(const details::log_msg &)> fun);
};

}  // namespace details
}  // namespace clog

#include <collie/log/details/backtracer-inl.h>
