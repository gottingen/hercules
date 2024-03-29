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

#include <collie/log/details/null_mutex.h>
#include <collie/log/details/synchronous_factory.h>
#include <collie/log/sinks/base_sink.h>

#include <mutex>
#include <string>

namespace clog {

// callbacks type
typedef std::function<void(const details::log_msg &msg)> custom_log_callback;

namespace sinks {
/*
 * Trivial callback sink, gets a callback function and calls it on each log
 */
template <typename Mutex>
class callback_sink final : public base_sink<Mutex> {
public:
    explicit callback_sink(const custom_log_callback &callback)
        : callback_{callback} {}

protected:
    void sink_it_(const details::log_msg &msg) override { callback_(msg); }
    void flush_() override{};

private:
    custom_log_callback callback_;
};

using callback_sink_mt = callback_sink<std::mutex>;
using callback_sink_st = callback_sink<details::null_mutex>;

}  // namespace sinks

//
// factory functions
//
template <typename Factory = clog::synchronous_factory>
inline std::shared_ptr<logger> callback_logger_mt(const std::string &logger_name,
                                                  const custom_log_callback &callback) {
    return Factory::template create<sinks::callback_sink_mt>(logger_name, callback);
}

template <typename Factory = clog::synchronous_factory>
inline std::shared_ptr<logger> callback_logger_st(const std::string &logger_name,
                                                  const custom_log_callback &callback) {
    return Factory::template create<sinks::callback_sink_st>(logger_name, callback);
}

}  // namespace clog
