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

#include "turbo/files/filesystem.h"
#include "turbo/log/details/null_mutex.h"
#include <turbo/log/sinks/base_sink.h>
#include "turbo/log/details/synchronous_factory.h"

#include <mutex>
#include <string>

namespace turbo::tlog {
    namespace sinks {
        /*
         * Trivial file sink with single file as target
         */
        template<typename Mutex>
        class basic_file_sink final : public base_sink<Mutex> {
        public:
            explicit basic_file_sink(const filename_t &filename, bool truncate = false,
                                     const turbo::FileEventListener &event_handlers = {});

            [[nodiscard]] filename_t filename() const;

        protected:
            void sink_it_(const details::log_msg &msg) override;

            void flush_() override;

        private:
            turbo::SequentialWriteFile file_writer_;
        };

        using basic_file_sink_mt = basic_file_sink<std::mutex>;
        using basic_file_sink_st = basic_file_sink<details::null_mutex>;

    } // namespace sinks

    //
    // factory functions
    //
    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger> basic_logger_mt(
            const std::string &logger_name, const filename_t &filename, bool truncate = false,
            const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::basic_file_sink_mt>(logger_name, filename, truncate, event_handlers);
    }

    template<typename Factory = turbo::tlog::synchronous_factory>
    inline std::shared_ptr<logger> basic_logger_st(
            const std::string &logger_name, const filename_t &filename, bool truncate = false,
            const turbo::FileEventListener &event_handlers = {}) {
        return Factory::template create<sinks::basic_file_sink_st>(logger_name, filename, truncate, event_handlers);
    }

} // namespace turbo::tlog

