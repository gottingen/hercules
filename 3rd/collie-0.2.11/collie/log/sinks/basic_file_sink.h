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

#include <collie/log/details/file_helper.h>
#include <collie/log/details/null_mutex.h>
#include <collie/log/details/synchronous_factory.h>
#include <collie/log/sinks/base_sink.h>

#include <mutex>
#include <string>

namespace clog {
namespace sinks {
/*
 * Trivial file sink with single file as target
 */
template <typename Mutex>
class basic_file_sink final : public base_sink<Mutex> {
public:
    explicit basic_file_sink(const filename_t &filename,
                             bool truncate = false,
                             const file_event_handlers &event_handlers = {});
    const filename_t &filename() const;

protected:
    void sink_it_(const details::log_msg &msg) override;
    void flush_() override;

private:
    details::file_helper file_helper_;
};

using basic_file_sink_mt = basic_file_sink<std::mutex>;
using basic_file_sink_st = basic_file_sink<details::null_mutex>;

}  // namespace sinks

//
// factory functions
//
template <typename Factory = clog::synchronous_factory>
inline std::shared_ptr<logger> basic_logger_mt(const std::string &logger_name,
                                               const filename_t &filename,
                                               bool truncate = false,
                                               const file_event_handlers &event_handlers = {}) {
    return Factory::template create<sinks::basic_file_sink_mt>(logger_name, filename, truncate,
                                                               event_handlers);
}

template <typename Factory = clog::synchronous_factory>
inline std::shared_ptr<logger> basic_logger_st(const std::string &logger_name,
                                               const filename_t &filename,
                                               bool truncate = false,
                                               const file_event_handlers &event_handlers = {}) {
    return Factory::template create<sinks::basic_file_sink_st>(logger_name, filename, truncate,
                                                               event_handlers);
}

}  // namespace clog

#include <collie/log/sinks/basic_file_sink-inl.h>
