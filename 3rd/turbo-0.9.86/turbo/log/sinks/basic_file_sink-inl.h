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

#include <turbo/log/sinks/basic_file_sink.h>
#include "turbo/log/common.h"
#include "turbo/log/details/os.h"

namespace turbo::tlog {
    namespace sinks {

        template<typename Mutex>
        basic_file_sink<Mutex>::basic_file_sink(const filename_t &filename,bool truncate,
                                                const turbo::FileEventListener &event_handlers)
                : file_writer_(event_handlers) {
            auto &open_option = truncate ? kLogTruncateOpenOption : kLogAppendOpenOption;
            auto r = file_writer_.open(filename, open_option);
            if (!r.ok()) {
                throw_tlog_ex(r.to_string());
            }
        }

        template<typename Mutex>
        filename_t basic_file_sink<Mutex>::filename() const {
            return file_writer_.file_path().native();
        }

        template<typename Mutex>
        void basic_file_sink<Mutex>::sink_it_(const details::log_msg &msg) {
            memory_buf_t formatted;
            base_sink<Mutex>::formatter_->format(msg, formatted);
            auto r = file_writer_.write(formatted);
            if (!r.ok()) {
                throw_tlog_ex(r.to_string());
            }
        }

        template<typename Mutex>
        void basic_file_sink<Mutex>::flush_() {
            auto r = file_writer_.flush();
            if (!r.ok()) {
                throw_tlog_ex(r.to_string());
            }
        }

    } // namespace sinks
} // namespace turbo::tlog
