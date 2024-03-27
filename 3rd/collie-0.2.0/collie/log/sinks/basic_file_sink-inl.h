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


#include <collie/log/common.h>
#include <collie/log/details/os.h>

namespace clog {
namespace sinks {

template <typename Mutex>
inline basic_file_sink<Mutex>::basic_file_sink(const filename_t &filename,
                                                      bool truncate,
                                                      const file_event_handlers &event_handlers)
    : file_helper_{event_handlers} {
    file_helper_.open(filename, truncate);
}

template <typename Mutex>
inline const filename_t &basic_file_sink<Mutex>::filename() const {
    return file_helper_.filename();
}

template <typename Mutex>
inline void basic_file_sink<Mutex>::sink_it_(const details::log_msg &msg) {
    memory_buf_t formatted;
    base_sink<Mutex>::formatter_->format(msg, formatted);
    file_helper_.write(formatted);
}

template <typename Mutex>
inline void basic_file_sink<Mutex>::flush_() {
    file_helper_.flush();
}

}  // namespace sinks
}  // namespace clog
